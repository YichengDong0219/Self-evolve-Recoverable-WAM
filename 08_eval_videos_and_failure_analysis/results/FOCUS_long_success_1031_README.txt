Focus: eval rollouts that SUCCEEDED after more than 400 steps (Claude, 2026-10-31, user request).
Searched every hanging_mug eval on the cluster (base, E 1000/2000/3000/3194, Mix 1000/2000/3000; >600 successes):
only ONE qualifies - Mix (mugM_nv) step 3000, seed 160082, success at step 551. All other successes end by step 374.

Timeline (gripper commands from executed_actions; video frame n = step n, 25 fps):
  62 / 131   left gripper close / open      (pick + hand-over, normal)
  225 / 334  right gripper close / open     first hang attempt -> mug falls off the rack onto the table (~340-420)
  473        right gripper closes again     re-grasps the fallen mug from the table
  546        right gripper opens            hung; SUCCESS at 551
Same seed elsewhere: E step 3000 does the same first attempt (221/322) and never re-grasps (fails at 900);
Mix step 2000 and 1000 re-grasp 4 times each but never hang it (fail at 900).

Files
  M3000_seed160082_SUCCESS_551steps.mp4                    full episode (head | left wrist | right wrist)
  M3000_seed160082_recovery_segment_step300-551.mp4        from the failed first hang to success
  M3000_seed160082_recovery_segment_step300-551_halfspeed.mp4
  M3000_seed160082_keyframes_step300-340-420-480-520-551.png
Replay check: replay_outcome_matches = true (bit-exact action replay with p1_render.py).

Same-seed comparisons (added when rendered):
  compare_mugE_nv_step3000_*  expert-only step 3000 (FAIL @ 900)
  compare_mugM_nv_step2000_*  Mix step 2000 (FAIL @ 900, 4 re-grasps)
  compare_mugM_nv_step1000_*  Mix step 1000 (FAIL @ 900, 4 re-grasps)
