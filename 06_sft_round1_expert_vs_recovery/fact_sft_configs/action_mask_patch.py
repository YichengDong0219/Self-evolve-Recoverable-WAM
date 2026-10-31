"""Per-frame action-loss mask for the hanging_mug recovery arm (Claude, 2026-10-29). Imported by mug_sft_mask.py.

FACT gates the action loss per EPISODE (meta/failure_rollouts.jsonl: failure_episode -> action_loss_mask 0 on every
frame; failure_active_from_frame -> value penalty). The recovery segments start with the robot holding still while the
fallen mug rocks to rest (median 53 frames, up to 401). Agreed with the user 10-29: before curobo's first motion only
the world-model losses train (video, future state, value without penalty); from the first motion on, the action loss
too. A failure_rollouts row {failure_episode: false, action_masked_until_frame: m} marks such an episode; this patch
treats frames with local index < m as failure_episode=True, failure_active=False, i.e. action_loss_mask 0, no value
penalty. Rows without the field and episodes without a row are unchanged (no-op for the E arm and the prefixes).
"""
import numpy as np
from fact_datasets.datasets.lerobot_dataset import LeRobotDataset

if not getattr(LeRobotDataset, "_action_mask_patch", False):
    _orig = LeRobotDataset._get_local_v2_data

    def _get_local_v2_data(self, index):
        d = _orig(self, index)
        row = self.failure_rows_by_episode.get(int(d["episode_index"]))
        if row is not None and not row.get("failure_episode", True) and "action_masked_until_frame" in row:
            if int(d["local_frame_index"]) < int(row["action_masked_until_frame"]):
                d["failure_episode"] = np.bool_(True)
                d["failure_active"] = np.bool_(False)
        return d

    LeRobotDataset._get_local_v2_data = _get_local_v2_data
    LeRobotDataset._action_mask_patch = True
