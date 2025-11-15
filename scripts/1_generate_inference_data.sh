
# 2. Random-M


# python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
#     --optimizer ProximalGradientDescent \
#     --device "cuda:0" \
#     --input-dim 32 \
#     --output-dim 25 \
#     --save-dir inference/QP3225 \
#     --save-to-mat \
#     --save-sol \
#     --optimizee-dir ./optimizees/matdata/qp-rand-3225


# python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
#     --optimizer ProximalGradientDescent \
#     --device "cuda:0" \
#     --input-dim 512 \
#     --output-dim 400 \
#     --save-dir inference/QP512400 \
#     --save-to-mat \
#     --save-sol \
#     --optimizee-dir ./optimizees/matdata/qp-rand-512400


# 2. Fixed-M for LISTA Inference


python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
    --optimizer ProximalGradientDescent \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --save-dir training/fixedM/QP512400-FixedM \
    --save-to-mat \
    --save-sol \
    --fixed-dict \
    --optimizee-dir ./optimizees/matdata/qp-rand-512400-fixedM