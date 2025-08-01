# SpaDec

## 环境配置
```shell
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## D-LLM

可参考D-LLM(D-LLM: A Token Adaptive Computing Resource Allocation Strategy for Large Language Models)的README，在此列出关键步骤

### 训练

```bash
cd src/LayerSkip/D_LLM/
bash finetuning_vicuna.sh
```

### 保存模型

1. 将最后的lora checkpoint复制到模型根目录下
2. `bash vicuna/inference_vicuna.sh`
3. 按照Vicuna-7b-dllm-temp的格式组织：
   1. `mv ./consolidated.00.pth /spadec/models/d-llm/Vicuna-7b-dllm-temp/`
   2. `cp /home/models/vicuna-7b-v1.5/tokenizer.model /spadec/models/d-llm/Vicuna-7b-dllm-temp/`
   3. `cp $OUTPUT_PATH$/model_args.json /spadec/models/d-llm/Vicuna-7b-dllm-temp/`
   4. `cp $OUTPUT_PATH$/params.json /spadec/models/d-llm/Vicuna-7b-dllm-temp/`

### 将D-LLM模型转换为huggingface格式：

```bash
python llama_hf/convert_llama_dllm_weights_to_hf.py \
    --input_dir /spadec/models/d-llm/Vicuna-7b-dllm-temp/ \
    --model_size 7B \
    --output_dir /spadec/models/d-llm/Vicuna-7b-dllm-temp-hf \
    --llama_version 2
# 手动删除tmp文件夹
```

为了适配Multi-token robust router，在src/LayerSkip/D_LLM/vicuna/model_train.py和src/LayerSkip/D_LLM/model.py中取消掉关于`x = self.asymmetric_quantization_for_similarity(x, quant_bits=4)`的注释

## EAGLE

可参考EAGLE(EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees)的README，在此列出关键步骤

### 生成训练数据

```bash
python -m eagle.ge_data.allocation --outdir [path of data]
```

### 训练EAGLE Head

```bash
accelerate launch -m --mixed_precision=bf16 eagle.train.main --tmpdir [path of data]\
--cpdir [path of checkpoints] --configpath [path of config file]
```

或者使用deepspeed：

```bash
bash train_eagle_deepspeed.sh
```

## Evaluation

Vicuna1.5-7B模型的Predictor权重已提供在`models/predictor/Vicuna_ours_predictor/`中

在src/SpecDecoding/EAGLE/eagle/model/ea_model.py中设置has_predictor=False以及True可以指定是否使用Predictor模块

运行测试：

```bash
python test/speed_example.py --base_model_path "/path/to/base/model" --ea_model_path "/path/to/ea/model" --save_path "/path/to/save/result.json"
python test/generate_example.py --path1 result1.json --path2 result2.json
```

