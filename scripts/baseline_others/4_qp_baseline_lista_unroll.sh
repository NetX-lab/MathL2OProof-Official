python main_unroll_listas.py --config ./configs/1_qp_training_train_anlys.yaml \
    --optimizer LISTACPSSWOnly \
    --optimizee-type QuadraticUnconstrained \
    --input-dim 512 \
    --output-dim 400 \
    --train-batch-size 2048 \
    --val-size 2048 \
    --init-lr 5e-4 \
    --optimizer-training-steps 100    --unroll-length 100   --layers 100 \
    --train-size 50 \
    --global-training-steps 50 \
    --pb 0.1 \
    --rho 0.2 \
    --lamb 0.4 \
    --identical-dict \
    --fixed-dict \
    --save-dir training/FixedM/LISTA-CPSS-WOnly-NotShared-lr0.0005-MInitW-UnrollTrain \
    --device "cuda:1"


# Inference
python main_unroll_listas.py --config ./configs/2_qp_testing.yaml \
    --optimizer LISTACPSSWOnly \
    --optimizee-type QuadraticUnconstrained \
    --input-dim 512 \
    --output-dim 400 \
    --train-batch-size 32 \
    --val-size 32 \
    --test-length 5000 \
    --test-size 32 \
    --test-batch-size 32 \
    --pb 0.1 \
    --rho 0.2 \
    --lamb 0.4 \
    --save-dir training/FixedM/LISTA-CPSS-WOnly-NotShared-lr0.0005-MInitW-UnrollTrain \
    --device "cuda:1"