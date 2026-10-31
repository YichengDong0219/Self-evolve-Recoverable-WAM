# Local (NJURL-8-3090) environment for FACT takeover iteration (Claude, 2026-10-28). Usage: source env.sh <gpu>
# Code is a copy of the remote RoboTwin-FACT @0aeea2d + probe code; assets link to ~/RoboTwin/assets (md5-identical
# for mug, rack, stapler, switch, aloha-agilex except absolute paths in curobo_*.yml).
L=/workspace/fact_local
export PATH=/workspace/miniconda3/envs/RoboTwin/bin:/usr/local/cuda-12.4/bin:$PATH
# FACT_ARCH: GPU compute capability for cuRobo's JIT kernels (3090 = 8.6 here; 4090 = 8.9 on the Amax server)
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=${FACT_ARCH:-8.6} TORCH_EXTENSIONS_DIR=$L/.torch_extensions${FACT_ARCH:+_$FACT_ARCH}
# CUDA_DEVICE_ORDER: make $1 mean the nvidia-smi index (Amax mixes RTX PRO 6000 and 4090; default FASTEST_FIRST reorders)
# pin by UUID (10-29): when a card drops out of CUDA (Amax 81:00/A1:00 "requires reset"), CUDA indices shift and
# an index would silently land on another card; nvidia-smi indices stay stable, so resolve index -> UUID here
_U=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader 2>/dev/null)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${_U:-$1} PYTHONUNBUFFERED=1 FACT_PROBE_DIR=$L/code
export PYTHONPATH=$L/FACT:$L/RoboTwin-FACT:$L/code
cd $L/RoboTwin-FACT
