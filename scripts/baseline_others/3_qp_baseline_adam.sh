# Case 1 Scale 3225
# python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
#     --optimizee-dir ./optimizees/matdata/qp-rand-3225 \
#     --optimizer Adam \
#     --input-dim 32 \
#     --output-dim 25 \
#     --step-size 1e-2 \
#     --momentum1 0.9 --momentum2 0.999 \
#     --test-length 5000 \
#     --load-mat \
#     --save-dir inference/Adam3225 \
#     --device "cuda:1"

# Case 1 Scale 512400
python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
    --optimizee-dir ./optimizees/matdata/qp-rand-512400 \
    --optimizer Adam \
    --input-dim 512 \
    --output-dim 400 \
    --momentum1 0.7 --momentum2 0.7 \
    --test-length 5000 \
    --load-mat \
    --loss-save-path qp-rand-beta1-0.9-beta2-0.7 \
    --save-dir inference/Adam512400 \
    --device "cuda:1"

# --step-size 1e-2 \
# python main_train_anlys.py --config ./configs/2_qp_testing.yaml --load-mat --optimizee-dir ./optimizees/matdata/lasso-rand  --optimizer AdamHD --step-size 0.1 --momentum1 0.001 --momentum2 0.1 --hyper-step 1e-07 --save-dir LASSO-AdamHD --device "cuda:0"