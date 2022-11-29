# Language Models for Novelty Detection in Kernel Traces

Description of the project

If you encounter an error, switch off your monitor, then you wont see it anymore.

# How to collect data

1. Deploy the server: [setup-server.md](resources/setup-server.md)

2. Deploy the client: [setup-client.md](resources/setup-client.md)

3. OPTIONAL Update the [LTTng](https://lttng.org) headers if necessary: [update-header.md](resources/update-header.md)

4. Collect the trace: [trace.md](resources/trace.md)

5. Generating OOD behaviors [setup-ood.md](resources/setup-ood.md)

A toy dataset is available in `data/toy` and a large dataset is available at [https://zenodo.org/record/7378420](https://zenodo.org/record/7378420).

# How to Analyse the Trace

1. Deploy the analysis environment:
    - Locally [setup-analysis.md](resources/setup-analysis.md)
    - On Compute Canada [setup-graham.md](resources/setup-graham.md)

2. Execute the experiment script:
    ```bash
    bash experiments.sh
    ```

# How to run the analysis

### N-gram with the datasets generation and statistics
```bash
python main.py --log_folder logs/bigram --data_path data/toy --train_folder "Train:train_id" --valid_id_folder "Valid ID:valid_id" --test_id_folder "Test ID:test_id" --valid_ood_folder "Valid OOD (Connection):valid_ood_connection,Valid OOD (CPU):valid_ood_cpu,Valid OOD (IO):valid_ood_dumpio,Valid OOD (OPCache):valid_ood_opcache,Valid OOD (Socket):valid_ood_socket,Valid OOD (SSL):valid_ood_ssl" --test_ood_folder "Test OOD (Connection):test_ood_connection,Test OOD (CPU):test_ood_cpu,Test OOD (IO):test_ood_dumpio,Test OOD (OPCache):test_ood_opcache,Test OOD (Socket):test_ood_socket,Test OOD (SSL):test_ood_ssl" --generate_dataset --dataset_stat --model ngram --order 3 --analysis
```

### LSTM
```bash
python main.py --log_folder logs/lstm --data_path data/toy --train_folder "Train:train_id" --valid_id_folder "Valid ID:valid_id" --test_id_folder "Test ID:test_id" --valid_ood_folder "Valid OOD (Connection):valid_ood_connection,Valid OOD (CPU):valid_ood_cpu,Valid OOD (IO):valid_ood_dumpio,Valid OOD (OPCache):valid_ood_opcache,Valid OOD (Socket):valid_ood_socket,Valid OOD (SSL):valid_ood_ssl" --test_ood_folder "Test OOD (Connection):test_ood_connection,Test OOD (CPU):test_ood_cpu,Test OOD (IO):test_ood_dumpio,Test OOD (OPCache):test_ood_opcache,Test OOD (Socket):test_ood_socket,Test OOD (SSL):test_ood_ssl" --max_token 8192 --model lstm --n_hidden 256 --n_layer 2 --dim_sys 48 --dim_proc 48 --dim_entry 12 --dim_ret 12 --dim_pid 12 --dim_tid 12 --dim_time 12 --dim_order 12 --optimizer adam --n_update 100000 --eval 100 --lr 0.001 --ls 0.1 --dropout 0.1 --batch 32 --gpu "0" --amp --reduce_lr_patience 5 --early_stopping_patience 10 --clip 10 --analysis
```

### Transformer with 2 GPUs
```bash
python main.py --log_folder logs/transformer --data_path data/toy --train_folder "Train:train_id" --valid_id_folder "Valid ID:valid_id" --test_id_folder "Test ID:test_id" --valid_ood_folder "Valid OOD (Connection):valid_ood_connection,Valid OOD (CPU):valid_ood_cpu,Valid OOD (IO):valid_ood_dumpio,Valid OOD (OPCache):valid_ood_opcache,Valid OOD (Socket):valid_ood_socket,Valid OOD (SSL):valid_ood_ssl" --test_ood_folder "Test OOD (Connection):test_ood_connection,Test OOD (CPU):test_ood_cpu,Test OOD (IO):test_ood_dumpio,Test OOD (OPCache):test_ood_opcache,Test OOD (Socket):test_ood_socket,Test OOD (SSL):test_ood_ssl" --max_token 1024 --model transformer --n_hidden 672 --n_layer 2 --n_head 4 --dim_sys 48 --dim_proc 48 --dim_entry 12 --dim_ret 12 --dim_pid 12 --dim_tid 12 --dim_time 12 --dim_order 12 --activation "swiglu" --optimizer adam --n_update 100000 --eval 100 --lr 0.001 --warmup_steps 1000 --ls 0.1 --dropout 0.1 --batch 16 --gpu "0,1" --chk --amp --reduce_lr_patience 5 --early_stopping_patience 10 --analysis
```

### Longformer
```bash
python main.py --log_folder logs/longformer --data_path data/toy --train_folder "Train:train_id" --valid_id_folder "Valid ID:valid_id" --test_id_folder "Test ID:test_id" --valid_ood_folder "Valid OOD (Connection):valid_ood_connection,Valid OOD (CPU):valid_ood_cpu,Valid OOD (IO):valid_ood_dumpio,Valid OOD (OPCache):valid_ood_opcache,Valid OOD (Socket):valid_ood_socket,Valid OOD (SSL):valid_ood_ssl" --test_ood_folder "Test OOD (Connection):test_ood_connection,Test OOD (CPU):test_ood_cpu,Test OOD (IO):test_ood_dumpio,Test OOD (OPCache):test_ood_opcache,Test OOD (Socket):test_ood_socket,Test OOD (SSL):test_ood_ssl" --valid_ood_folder "Valid OOD (CPU):valid_ood_cpu" --test_ood_folders "Test OOD (CPU):valid_ood_cpu" --max_token 8192 --model longformer --n_hidden 672 --n_layer 2 --n_head 4 --window "32,64" --dilatation "1,1" --global_att "0" --dim_sys 48 --dim_proc 48 --dim_entry 12 --dim_ret 12 --dim_pid 12 --dim_tid 12 --dim_time 12 --dim_order 12 --activation "gelu" --optimizer adam --n_update 100000 --eval 100 --lr 0.001 --warmup_steps 1000 --ls 0.1 --dropout 0.1 --batch 32 --gpu "0" --amp --reduce_lr_patience 5 --early_stopping_patience 10 --analysis
```
