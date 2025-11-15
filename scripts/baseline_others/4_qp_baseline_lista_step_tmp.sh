

# Single Opt-coefficient matrix M, learnable parameter W initialized with M
python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --optimizer LISTACPSSSTEP \
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
    --save-dir training/FixedM/LISTA-CPSS-Step-Shared-lr0.001 \
    --device "cuda:0"
    
# Multiple Opt-coefficient matrix M, learnable parameter W initialized with M-mean
python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --optimizer LISTACPSSSTEP \
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
    --save-dir training/MultipleM/LISTA-CPSS-Step-Shared-lr0.001 \
    --device "cuda:0"

