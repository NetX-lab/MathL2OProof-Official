# NOTE Case Scale 3225

# 1. Generate QP instances and save them to "./optimizees/matdata/qp-rand-3225"

# python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
#     --optimizer ProximalGradientDescent \
#     --device "cuda:0" \
#     --input-dim 32 \
#     --output-dim 25 \
#     --save-dir inference/QP3225 \
#     --save-to-mat \
#     --save-sol \
#     --optimizee-dir ./optimizees/matdata/qp-rand-3225

# 2. Use GD to generate 
# python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
#     --optimizee-dir ./optimizees/matdata/qp-rand-3225 \
#     --optimizer ProximalGradientDescent \
#     --input-dim 32 \
#     --output-dim 25 \
#     --test-length 5000 \
#     --load-mat \
#     --save-dir inference/GD3225 \
#     --device "cuda:0"
    


# NOTE Case Scale 512400
# python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
#     --optimizer ProximalGradientDescent \
#     --device "cuda:0" \
#     --input-dim 512 \
#     --output-dim 400 \
#     --save-dir inference/QP512400 \
#     --save-to-mat \
#     --save-sol \
#     --optimizee-dir ./optimizees/matdata/qp-rand-512400

# 2. Use GD to generate 
python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
    --optimizee-dir ./optimizees/matdata/qp-rand-512400 \
    --optimizer ProximalGradientDescent \
    --input-dim 512 \
    --output-dim 400 \
    --test-length 5000 \
    --load-mat \
    --save-dir inference/GD512400 \
    --device "cuda:0"