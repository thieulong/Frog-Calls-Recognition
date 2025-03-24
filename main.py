import os
import json
import collections
import numpy as np
import pandas as pd
from etils import epath
import tensorflow as tf
from tqdm import tqdm
from ml_collections import config_dict
from chirp import audio_utils
from chirp.inference import tf_examples, embed_lib
from chirp.inference.search import bootstrap
from chirp.inference.classify import classify
from perch_hoplite.zoo import zoo_interface, model_configs

# Load configuration
with open("config.json") as f:
    config_data = json.load(f)

input_dir = epath.Path(config_data['input_dir'])
output_dir = epath.Path(config_data['output_dir'])
model_dir = epath.Path(config_data['model_dir'])
working_dir = epath.Path(config_data['working_dir'])

embeddings_path = working_dir / "embeddings"
embeddings_path.mkdir(parents=True, exist_ok=True)

# Embedding Configuration
model_choice = config_data['model_choice']
preset_info = model_configs.get_preset_model_config(model_choice)

config = config_dict.ConfigDict()
config.embed_fn_config = config_dict.ConfigDict()
config.embed_fn_config.model_config = preset_info.model_config
config.embed_fn_config.model_key = preset_info.model_key
config.embed_fn_config.write_embeddings = True
config.embed_fn_config.write_logits = False
config.embed_fn_config.write_separated_audio = False
config.embed_fn_config.write_raw_audio = False
config.embed_fn_config.file_id_depth = 1
config.source_file_patterns = [str(input_dir / "*")]
config.output_dir = embeddings_path.as_posix()

embed_fn = embed_lib.EmbedFn(**config.embed_fn_config)
embed_fn.min_audio_s = 1.0
embed_fn.setup()

# Prepare audio sources
source_infos = embed_lib.create_source_infos(
    config.source_file_patterns,
    num_shards_per_file=config.get('num_shards_per_file', -1),
    shard_len_s=config.get('shard_len_s', -1))

existing_embedding_ids = embed_lib.get_existing_source_ids(embeddings_path, 'embeddings-*')
new_source_infos = embed_lib.get_new_source_infos(source_infos, existing_embedding_ids, config.embed_fn_config.file_id_depth)

# Generate embeddings
with tf_examples.EmbeddingsTFRecordMultiWriter(output_dir=embeddings_path, num_files=1) as file_writer:
    audio_iterator = audio_utils.multi_load_audio_window(
        filepaths=[s.filepath for s in new_source_infos],
        offsets=[s.shard_num * s.shard_len_s for s in new_source_infos],
        audio_loader=lambda fp, offset: audio_utils.load_audio_window(
            fp, offset, sample_rate=config.embed_fn_config.model_config.sample_rate,
            window_size_s=config.get('shard_len_s', -1.0)),
    )
    for source_info, audio in tqdm(zip(new_source_infos, audio_iterator), total=len(new_source_infos)):
        if not embed_fn.validate_audio(source_info, audio):
            continue
        file_id = source_info.file_id(config.embed_fn_config.file_id_depth)
        offset_s = source_info.shard_num * source_info.shard_len_s
        example = embed_fn.audio_to_example(file_id, offset_s, audio)
        if example:
            file_writer.write(example.SerializeToString())

# Write minimal config.json required for bootstrap
embedding_config_path = embeddings_path / "config.json"
with embedding_config_path.open("w") as f:
    json.dump({
        "sample_rate": preset_info.model_config.sample_rate,
        "embedding_hop_size_s": getattr(preset_info.model_config, 'embedding_stride_s', 0.5),
        "embedding_size": getattr(preset_info.model_config, 'embedding_size', 2048),
        "window_size_s": getattr(preset_info.model_config, 'window_size_s', 5.0),
        "source_file_patterns": [str(input_dir / "*")],
        "embed_fn_config": {
            "file_id_depth": 1,
            "model_key": preset_info.model_key,
            "model_config": {
                "sample_rate": preset_info.model_config.sample_rate,
                "embedding_stride_s": getattr(preset_info.model_config, 'embedding_stride_s', 0.5),
                "embedding_size": getattr(preset_info.model_config, 'embedding_size', 2048),
                "window_size_s": getattr(preset_info.model_config, 'window_size_s', 5.0)
            }
        }
    }, f)

# Inference functions
def run_inference(species, species_model_dir, embeddings_path, threshold):
    print("---")
    print(f"\n[INFO] Running inference for species: {species}")
    labeled_data_path = epath.Path(species_model_dir) / 'labeled'
    subdirs = [d.name for d in labeled_data_path.iterdir() if d.is_dir() and not d.name.startswith('embeddings-')]
    detect_class = subdirs[0] if subdirs else species.lower().replace(' ', '_')

    custom_classifier_path = epath.Path(species_model_dir) / 'custom_classifier'
    bootstrap_config = bootstrap.BootstrapConfig.load_from_embedding_path(
        embeddings_path=embeddings_path,
        annotated_path=labeled_data_path)
    cfg = config_dict.ConfigDict({
        'model_path': custom_classifier_path,
        'logits_key': 'custom',
    })
    logits_head = zoo_interface.LogitsOutputHead.from_config(cfg)
    print(f"[INFO] Loaded custom model with classes: \n - " + "\n - ".join(logits_head.class_list.classes))

    embeddings_ds = tf_examples.create_embeddings_dataset(embeddings_path, file_glob='embeddings-*')
    output_file = working_dir / f"inference-{species}.csv"
    threshold_map = collections.defaultdict(lambda: threshold)

    classify.write_inference_csv(
        embeddings_ds=embeddings_ds,
        model=logits_head,
        labels=logits_head.class_list.classes,
        output_filepath=output_file,
        threshold=threshold_map,
        embedding_hop_size_s=bootstrap_config.embedding_hop_size_s,
        include_classes=[detect_class],
        exclude_classes=['unknown'])

    # Print detection summary
    df = pd.read_csv(output_file)
    if 'label' in df.columns:
        detections = (df['label'] == detect_class).sum()
        nondetections = len(df) - detections
        print(f"[INFO] Results:")
        print(f"   Detection count:  {detections}")
        print(f"NonDetection count:  {nondetections}")
        print("---")
    print(f"[INFO] Inference completed. Results saved to: {output_file}\n")
    return output_file

def format_results(inference_files, output_csv_path):
    all_dfs = []
    for file in inference_files:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df.rename(columns={'timestamp_s': 'start'}, inplace=True)
        df['end'] = df['start'] + 5.0
        df['filename'] = df['filename'].str.replace('data/', '', regex=False)
        df['accuracy'] = 1 / (1 + np.exp(-df['logit']))
        df = df[['filename', 'start', 'end', 'label', 'accuracy']]
        all_dfs.append(df)

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        merged.to_csv(output_csv_path, index=False)
        print(f"[INFO] Final results written to: {output_csv_path}")
    else:
        print("[WARN] No results to merge.")

# Run inference
species_list = [s for s in os.listdir(model_dir) if os.path.isdir(model_dir / s) and s != 'embeddings']
inference_outputs = []

if config_data['detect_species'].lower() == 'all species':
    for species in species_list:
        path = model_dir / species
        inference_outputs.append(run_inference(species, path, embeddings_path, config_data['default_threshold']))
    result_file = output_dir / "results-all.csv"
else:
    species = config_data['detect_species']
    path = model_dir / species
    inference_outputs.append(run_inference(species, path, embeddings_path, config_data['default_threshold']))
    result_file = output_dir / f"results-{species}.csv"

format_results(inference_outputs, result_file)
