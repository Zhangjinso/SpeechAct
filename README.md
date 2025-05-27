 **SpeechAct: Towards Generating Whole-body Motion from Speech** 


1.Setup environment
Create conda environment:

```
conda env create -f environment.yml
conda activate speechact
```
2.Test
Download [pretrained models](https://drive.google.com/file/d/1FPalJ3NK5EY_kzmBa6LChz2vN48ZNbSZ/view?usp=drive_link), Extract to the current folder:

```
python demo.py --infer --audio_file test.wav --speaker_names scott
```



