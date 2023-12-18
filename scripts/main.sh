
#sh scripts/eval.sh rel P01_04 prova 'summary' 0 0
#sh scripts/eval.sh rel P01_01 rel 'masks' 0 0
#sh scripts/eval.sh rel P01_01 rel 'SingleFrames' 0 0
#sh scripts/eval.sh rel P01_01 rel 'SingleMasks' 0 0
CKP=ckpts/$1
VID=$2
EXP=$3
OUT=$4
MASKS_N_SAMPLES=$5
SUMMARY_N_SAMPLES=$6

# Define your list of values
th_values=(0.1 0.12 0.15 0.17 0.2 0.22 0.25 0.3 0.32 0.35 0.4 0.6 1)

# Loop through each value and print it
for val in "${th_values[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python colmap_visualizers.py \
  --root "/Users/francesco/Desktop/Università/Tesi/EgoChiara/CodiceEPICFIELDS/depth_extractor/data/Epic_converted"\
  --vid "P01_01" \
  --operation "Dense" \
  --threshold_dist $val \
  --segment_th 2 \
done



  
