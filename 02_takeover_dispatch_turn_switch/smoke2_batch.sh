#!/bin/bash
# Second takeover smoke (Claude, 2026-10-27): stock expert on turn_switch / move_stapler_pad P1 failures; grasp scan on mug.
D=~/fact_g0_claude/code
for x in "P1_turn_switch_demo_clean_00 96" "P1_turn_switch_demo_clean_02 96" "P1_turn_switch_demo_clean_03 96" "P1_turn_switch_demo_clean_04 96" \
         "P1_turn_switch_demo_clean_06 96" "P1_turn_switch_demo_clean_08 96" "P1_turn_switch_demo_clean_09 96" "P1_turn_switch_demo_clean_13 288" \
         "P1_turn_switch_demo_clean_18 96" "P1_move_stapler_pad_demo_clean_04 192" "P1_move_stapler_pad_demo_clean_07 192" \
         "P1_move_stapler_pad_demo_clean_10 192" "P1_move_stapler_pad_demo_clean_16 192"; do
  set -- $x; echo "== takeover $1 @$2"; bash $D/run_takeover.sh 1 $1 A $2
done
for i in 00 01 02 03 04 06 09 11 12 13 14 16 17 19; do echo "== scan mug $i"; bash $D/run_scan.sh 1 P1_hanging_mug_demo_clean_$i 336; done
echo ALL_DONE
