# [generating test instances and save them]

## generate LASSO instances and save them to "./optimizees/matdata/lasso-rand"
# python main.py --config ./configs/2_lasso_testing.yaml --optimizer ProximalGradientDescentMomentum --save-dir LASSO-FISTA --save-to-mat --optimizee-dir ./optimizees/matdata/lasso-rand

## solve LASSO with FISTA and save the optimal objective value for each instance (5000 iterations are sufficient to obtain optimal objective)
# python main.py --config ./configs/2_lasso_testing.yaml --optimizer ProximalGradientDescentMomentum --save-dir LASSO-FISTA --load-mat --save-sol --optimizee-dir ./optimizees/matdata/lasso-rand --test-length 5000


# [train and test models for Ada-LISTA] //This may take long time for problems with size of 250*500.
# python main_unroll.py --optimizer AdaLISTA \
#     --optimizee-type LASSO \
#     --input-dim 500 --sparsity 50 --output-dim 250 \
#     --layers 10 --init-lr 2e-3 --save-dir LASSO-AdaLISTA --device "cuda:0"

# python main_unroll.py --config ./configs/2_lasso_testing.yaml \
#     --load-mat --load-sol --optimizee-dir ./optimizees/matdata/lasso-rand \
#     --optimizer AdaLISTA --layers 10 --init-lr 2e-3 --save-dir LASSO-AdaLISTA \
#     --device "cuda:0" --test-batch-size 4

# [train and test models for LISTA] W shared
python main_unroll_listas.py --optimizer LISTA \
    --optimizee-type LASSO_LISTA \
    --input-dim 500 --sparsity 50 --output-dim 250 \
    --layers 100 --init-lr 1e-7 --save-dir baseline/LASSO-LISTA \
    --train-size 32000 \
    --train-batch-size 256 \
    --fixed-dict \
    --pb 0.1 \
    --rho 0.2 \
    --lamb 0.4 \
    --device "cuda:0"

# python main_unroll_listas.py --config ./configs/2_lasso_testing.yaml \
#     --optimizee-type LASSO_LISTA \
#     --optimizee-dir ./optimizees/matdata/lasso-rand \
#     --optimizer LISTA --layers 16 --save-dir baseline/LASSO-LISTA \
#     --fixed-dict \
#     --pb 0.1 \
#     --rho 0.2 \
#     --lamb 0.4 \
#     --test-batch-size 256 \
#     --eval-metric 'nmse' \
#     --loss-save-path lasso-rand-nmse \
#     --device "cuda:0"

# python main_unroll_listas.py --config ./configs/2_lasso_testing.yaml \
#     --optimizee-type LASSO_LISTA \
#     --optimizee-dir ./optimizees/matdata/lasso-rand \
#     --optimizer LISTA --layers 16 --save-dir baseline/LASSO-LISTA \
#     --fixed-dict \
#     --pb 0.1 \
#     --rho 0.2 \
#     --lamb 0.4 \
#     --test-batch-size 256 \
#     --eval-metric 'obj' \
#     --loss-save-path lasso-rand-obj \
#     --device "cuda:0" \
#     --debug


# [train and test models for LISTA] W not shared
# python main_unroll_listas.py --optimizer LISTA \
#     --optimizee-type LASSO_LISTA \
#     --input-dim 500 --sparsity 50 --output-dim 250 \
#     --layers 16 --init-lr 1e-7 --save-dir baseline/LASSO-LISTA-WnotShared \
#     --train-size 32000 \
#     --train-batch-size 256 \
#     --fixed-dict \
#     --pb 0.1 \
#     --rho 0.2 \
#     --lamb 0.4 \
#     --device "cuda:0"

# python main_unroll_listas.py --config ./configs/2_lasso_testing.yaml \
#     --optimizee-type LASSO_LISTA \
#     --optimizee-dir ./optimizees/matdata/lasso-rand \
#     --optimizer LISTA --layers 16 --save-dir baseline/LASSO-LISTA-WnotShared \
#     --fixed-dict \
#     --pb 0.1 \
#     --rho 0.2 \
#     --lamb 0.4 \
#     --test-batch-size 256 \
#     --eval-metric 'nmse' \
#     --loss-save-path lasso-rand-nmse \
#     --device "cuda:0"

# python main_unroll_listas.py --config ./configs/2_lasso_testing.yaml \
#     --optimizee-type LASSO_LISTA \
#     --optimizee-dir ./optimizees/matdata/lasso-rand \
#     --optimizer LISTA --layers 16 --save-dir baseline/LASSO-LISTA-WnotShared \
#     --fixed-dict \
#     --pb 0.1 \
#     --rho 0.2 \
#     --lamb 0.4 \
#     --test-batch-size 256 \
#     --eval-metric 'obj' \
#     --loss-save-path lasso-rand-obj \
#     --device "cuda:0" \
#     --debug