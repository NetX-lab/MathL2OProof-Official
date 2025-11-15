
# Not Fixed Parameter M
# python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
#     --optimizer LISTACPSS \
#     --optimizee-type QuadraticUnconstrained \
#     --input-dim 512 \
#     --output-dim 400 \
#     --train-batch-size 10 \
#     --init-lr 0.001 \
#     --optimizer-training-steps 100    --unroll-length 100 \
#     --global-training-steps 5000 \
#     --rho 0.2 \
#     --lamb 0.4 \
#     --w-shared \
#     --theta-shared \
#     --save-dir training/LISTA-CPSS-Shared-LR10-2 \
#     --device "cuda:0"


# # Fixed Parameter M
# python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
#     --optimizer LISTACPSS \
#     --optimizee-type QuadraticUnconstrained \
#     --input-dim 512 \
#     --output-dim 400 \
#     --train-batch-size 10 \
#     --init-lr 0.001 \
#     --optimizer-training-steps 100    --unroll-length 100 \
#     --global-training-steps 5000 \
#     --rho 0.2 \
#     --lamb 0.4 \
#     --fixed-dict \
#     --w-shared \
#     --theta-shared \
#     --save-dir training/LISTA-CPSS-Shared-FixM-InitMeanM-lr0.001 \
#     --device "cuda:0"


# Not Fixed Parameter M
# python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
#     --optimizer LISTACPSS \
#     --optimizee-type QuadraticUnconstrained \
#     --input-dim 512 \
#     --output-dim 400 \
#     --train-batch-size 10 \
#     --init-lr 0.0000001 \
#     --optimizer-training-steps 100    --unroll-length 100 \
#     --global-training-steps 5000 \
#     --rho 0.2 \
#     --lamb 0.4 \
#     --w-shared \
#     --theta-shared \
#     --save-dir training/LISTA-CPSS-Shared-RandInitW-lr0.0000001 \
#     --device "cuda:0"


# Fixed Parameter M
# python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
#     --optimizer LISTACPSS \
#     --optimizee-type QuadraticUnconstrained \
#     --input-dim 512 \
#     --output-dim 400 \
#     --train-batch-size 10 \
#     --init-lr 0.0000001 \
#     --optimizer-training-steps 100    --unroll-length 100 \
#     --global-training-steps 5000 \
#     --rho 0.2 \
#     --lamb 0.4 \
#     --fixed-dict \
#     --w-shared \
#     --theta-shared \
#     --save-dir training/LISTA-CPSS-Shared-FixM-RandInitW-lr0.0000001 \
#     --device "cuda:1"


# Single Opt-coefficient matrix M, learnable parameter W initialized with M
python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --optimizer LISTACPSS \
    --optimizee-type QuadraticUnconstrained \
    --input-dim 512 \
    --output-dim 400 \
    --train-batch-size 10 \
    --init-lr 0.001 \
    --optimizer-training-steps 100    --unroll-length 100 \
    --global-training-steps 5000 \
    --rho 0.2 \
    --lamb 0.4 \
    --identical-dict \
    --fixed-dict \
    --w-shared \
    --theta-shared \
    --save-dir training/FixedM/LISTA-CPSS-Shared-lr0.001 \
    --device "cuda:0"
    
# Multiple Opt-coefficient matrix M, learnable parameter W initialized with M-mean
python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --optimizer LISTACPSS \
    --optimizee-type QuadraticUnconstrained \
    --input-dim 512 \
    --output-dim 400 \
    --train-batch-size 10 \
    --init-lr 0.001 \
    --optimizer-training-steps 100    --unroll-length 100 \
    --global-training-steps 5000 \
    --rho 0.2 \
    --lamb 0.4 \
    --identical-dict \
    --w-shared \
    --theta-shared \
    --save-dir training/MultipleM/LISTA-CPSS-Shared-lr0.001 \
    --device "cuda:0"

