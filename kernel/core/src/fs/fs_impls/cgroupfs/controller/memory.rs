// SPDX-License-Identifier: MPL-2.0

use alloc::sync::Arc;
use core::sync::atomic::{AtomicUsize, Ordering};

use aster_systree::{Error, Result, SysAttrSetBuilder, SysPerms, SysStr};
use aster_util::printer::VmPrinter;
use ostd::{
    mm::{PAGE_SIZE, VmReader, VmWriter},
    sync::RcuOptionReadGuard,
    task::atomic_mode::AsAtomicModeGuard,
};

use crate::fs::cgroupfs::systree_node::{CgroupNode, CgroupSysNode, CgroupSystem};

/// A sub-controller responsible for memory resource management in the cgroup subsystem.
///
/// This controller currently performs *accounting only*: it tracks how much anonymous
/// memory is resident in the cgroup's subtree, and reports it through `memory.current`,
/// `memory.peak` and `memory.stat`.
///
/// Like the CPU sub-controller, this one is instantiated even while inactive. Charges are
/// applied to the levels of the hierarchy that hold a controller, so if a level could appear
/// or disappear between a page being charged and being uncharged, the uncharge would not
/// find the counter it incremented and would underflow. Staying instantiated makes every
/// charge and its matching uncharge see the same set of counters.
///
/// `memory.max` is deliberately **not** provided yet. A limit that is accepted but never
/// enforced looks like isolation while providing none, which is worse for a caller than an
/// absent knob: `memory.max` is the interface a container runtime probes to decide whether
/// it can rely on the kernel to cap a workload. It is added in the same change that adds
/// enforcement, not before. `memory.events` is omitted for the same reason -- every event it
/// counts is defined relative to a limit, so with no limits it can only ever report zeros.
pub(crate) struct MemoryController {
    /// The number of bytes of anonymous memory charged to this cgroup's subtree.
    current_bytes: AtomicUsize,
    /// The peak value ever observed for `current_bytes`.
    peak_bytes: AtomicUsize,
    /// Whether `+memory` is enabled for this cgroup.
    ///
    /// Accounting continues while inactive so that charges stay balanced, but none of the
    /// interface files are exposed. See [`MemoryController::is_attr_absent`].
    is_active: bool,
}

impl MemoryController {
    pub(super) fn init_attr_set(builder: &mut SysAttrSetBuilder, is_root: bool) {
        // These attributes only exist on the non-root cgroup nodes.
        // However, it seems that the `memory.stat` attribute is also present on the root node in practice.
        // Currently the implementation follows the documentation strictly.
        //
        // Reference: <https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory-interface-files>
        if !is_root {
            builder.add(
                SysStr::from("memory.current"),
                SysPerms::DEFAULT_RO_ATTR_PERMS,
            );
            builder.add(SysStr::from("memory.peak"), SysPerms::DEFAULT_RO_ATTR_PERMS);
            builder.add(SysStr::from("memory.stat"), SysPerms::DEFAULT_RO_ATTR_PERMS);
        }
    }

    /// Carries the accounted totals over from the sub-controller being replaced.
    ///
    /// Activating or deactivating `+memory` builds a fresh sub-controller for each child. The
    /// charges already applied must survive that, or a later uncharge would underflow.
    pub(super) fn init_stats(&mut self, previous: &Self) {
        *self.current_bytes.get_mut() = previous.current_bytes.load(Ordering::Relaxed);
        *self.peak_bytes.get_mut() = previous.peak_bytes.load(Ordering::Relaxed);
    }

    /// Charges `bytes` of anonymous memory to this cgroup.
    fn charge(&self, bytes: usize) {
        let new_bytes = self.current_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.peak_bytes.fetch_max(new_bytes, Ordering::Relaxed);
    }

    /// Uncharges `bytes` of anonymous memory from this cgroup.
    fn uncharge(&self, bytes: usize) {
        let old_bytes = self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
        debug_assert!(
            old_bytes >= bytes,
            "memory current count underflow: {} charged, {} uncharged",
            old_bytes,
            bytes
        );
    }
}

impl super::SubControl for MemoryController {
    fn is_attr_absent(&self, _name: &str) -> bool {
        // Unlike `cpu.stat`, no memory interface file is exposed before `+memory` is
        // enabled, which is what Linux does.
        !self.is_active
    }

    fn read_attr_at(&self, name: &str, offset: usize, writer: &mut VmWriter) -> Result<usize> {
        let mut printer = VmPrinter::new_skip(writer, offset);

        match name {
            "memory.current" => {
                let current = self.current_bytes.load(Ordering::Relaxed);
                writeln!(printer, "{}", current)?;
            }
            "memory.peak" => {
                let peak = self.peak_bytes.load(Ordering::Relaxed);
                writeln!(printer, "{}", peak)?;
            }
            "memory.stat" => {
                // Only the categories that are actually accounted are reported. Linux prints
                // many more, but printing a field that is hardcoded to zero would claim the
                // category is tracked and empty rather than untracked.
                let anon = self.current_bytes.load(Ordering::Relaxed);
                writeln!(printer, "anon {}", anon)?;
            }
            _ => return Err(Error::AttributeError),
        }

        Ok(printer.bytes_written())
    }

    fn write_attr(&self, _name: &str, _reader: &mut VmReader) -> Result<usize> {
        // Every attribute this controller provides is read-only.
        Err(Error::AttributeError)
    }
}

impl super::SubControlStatic for MemoryController {
    fn new(is_root: bool, is_active: bool) -> Self {
        Self {
            current_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            is_active: is_root || is_active,
        }
    }

    fn type_() -> super::SubCtrlType {
        super::SubCtrlType::Memory
    }

    fn read_from(controller: &super::Controller) -> Arc<super::SubController<Self>> {
        controller.memory.read().get().clone()
    }
}

/// Hierarchical memory charge/uncharge operations.
impl super::SubController<MemoryController> {
    /// Charges `pages` of anonymous memory across the hierarchy.
    ///
    /// The charge is applied at every level from this cgroup up to the root at which the
    /// memory sub-controller is active. There is no limit to check, so unlike
    /// [`super::SubController<PidsController>::try_charge_hierarchy`] this cannot fail and
    /// needs no rollback.
    ///
    /// [`super::SubController<PidsController>::try_charge_hierarchy`]: super::SubController
    fn charge_hierarchy(&self, pages: usize) {
        let bytes = pages * PAGE_SIZE;

        let mut current = Some(self);
        while let Some(node) = current {
            if let Some(ref inner) = node.inner {
                inner.charge(bytes);
            }
            current = node.parent.as_deref();
        }
    }

    /// Uncharges `pages` of anonymous memory across the hierarchy.
    fn uncharge_hierarchy(&self, pages: usize) {
        let bytes = pages * PAGE_SIZE;

        let mut current = Some(self);
        while let Some(node) = current {
            if let Some(ref inner) = node.inner {
                inner.uncharge(bytes);
            }
            current = node.parent.as_deref();
        }
    }
}

impl super::Controller {
    /// Charges `pages` of anonymous memory in the memory sub-controller hierarchy.
    fn charge_memory<G: AsAtomicModeGuard + ?Sized>(&self, guard: &G, pages: usize) {
        self.memory.read_with(guard).charge_hierarchy(pages);
    }

    /// Uncharges `pages` of anonymous memory in the memory sub-controller hierarchy.
    fn uncharge_memory<G: AsAtomicModeGuard + ?Sized>(&self, guard: &G, pages: usize) {
        self.memory.read_with(guard).uncharge_hierarchy(pages);
    }
}

/// Charges `pages` of anonymous memory to the cgroup hierarchy that `cgroup` refers to.
///
/// If `cgroup` refers to no cgroup, the charge is applied to the root cgroup, the same way
/// [`charge_cpu_time`] resolves a process that is not attached to a non-root cgroup.
///
/// [`charge_cpu_time`]: super::cpu::charge_cpu_time
pub(crate) fn charge_memory(cgroup: &RcuOptionReadGuard<'_, Arc<CgroupNode>>, pages: usize) {
    if let Some(node) = cgroup.get() {
        node.controller().charge_memory(cgroup, pages);
    } else {
        CgroupSystem::singleton()
            .controller()
            .charge_memory(cgroup, pages);
    }
}

/// Uncharges `pages` of anonymous memory from the cgroup hierarchy that `cgroup` refers to.
///
/// See [`charge_memory`] for how an unattached cgroup reference is resolved.
pub(crate) fn uncharge_memory(cgroup: &RcuOptionReadGuard<'_, Arc<CgroupNode>>, pages: usize) {
    if let Some(node) = cgroup.get() {
        node.controller().uncharge_memory(cgroup, pages);
    } else {
        CgroupSystem::singleton()
            .controller()
            .uncharge_memory(cgroup, pages);
    }
}
