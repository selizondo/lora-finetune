import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient

NB_PATH = r'c:\Users\selizondo\projects\selizondo\coding_labs\lora-finetune\train_colab.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=36000, kernel_name='python3')
client.execute()

with open(NB_PATH, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print('Notebook execution complete.')
