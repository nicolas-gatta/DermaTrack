from django import forms

class TrainingForm(forms.Form):
    train_file = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'id': 'trainFile',
            'placeholder': 'Enter path to train file',
            'required': 'required'
        }),
        label="Train File"
    )
    
    eval_file = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'id': 'evalFile',
            'placeholder': 'Enter path to eval file',
            'required': 'required'
        }),
        label="Eval File"
    )
    
    output_dir = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'id': 'outputDir',
            'placeholder': 'Enter output directory',
            'required': 'required'
        }),
        label="Output Directory"
    )
    
    learning_rate = forms.FloatField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'id': 'learningRate',
            'placeholder': '1e-4',
            'step': '0.0001',
            'value': '0.0001',
            'required': 'required'
        }),
        label="Learning Rate"
    )
    
    seed = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'id': 'seed',
            'placeholder': '1',
            'value': '1',
            'required': 'required'
        }),
        label="Seed"
    )
    
    batch_size = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'id': 'batchSize',
            'placeholder': '16',
            'value': '16',
            'required': 'required'
        }),
        label="Batch Size"
    )
    
    num_epochs = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'id': 'numEpochs',
            'placeholder': '100',
            'value': '100',
            'required': 'required'
        }),
        label="Number of Epochs"
    )
    
    num_workers = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'id': 'numWorkers',
            'placeholder': '8',
            'value': '8',
            'required': 'required'
        }),
        label="Number of Workers"
    )
