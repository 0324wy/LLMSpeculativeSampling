from huggingface_hub import snapshot_download

print('begin')
snapshot_download(repo_id="bigscience/bloom-560m")