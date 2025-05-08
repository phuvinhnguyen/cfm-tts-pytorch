from datasets import load_dataset
from torch.utils.data import Dataset

class LJSpeech(Dataset):
    '''
    Each sample of the LJSpeech:
    {
        'id': 'LJ002-0026',
        'file': '/datasets/downloads/extracted/05bfe561f096e4c52667e3639af495226afe4e5d08763f2d76d069e7a453c543/LJSpeech-1.1/wavs/LJ002-0026.wav',
        'audio': {'path': '/datasets/downloads/extracted/05bfe561f096e4c52667e3639af495226afe4e5d08763f2d76d069e7a453c543/LJSpeech-1.1/wavs/LJ002-0026.wav',
        'array': array([-0.00048828, -0.00018311, -0.00137329, ...,  0.00079346,
                0.00091553,  0.00085449], dtype=float32),
        'sampling_rate': 22050},
        'text': 'in the three years between 1813 and 1816,'
        'normalized_text': 'in the three years between eighteen thirteen and eighteen sixteen,',
    }
    '''
    def __init__(self):
        self.dataset = load_dataset("keithito/lj_speech")
        # Convert normalized_text to transcript
        # Convert audio to {array, sampling_rate}
        self.dataset.rename_column('normalized_text', 'transcript')
        self.dataset['audio'] = {
            'array': self.dataset['audio']['array'],
            'sampling_rate': self.dataset['audio']['sampling_rate']
        }

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    