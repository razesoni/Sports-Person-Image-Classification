# Sports Person Image Classification

A notebook exploring face cropping, raw pixel features, wavelet features and classical image classifiers. It compares SVM, Random Forest and Logistic Regression with training-set grid search.

## Run locally

From the repository root in a separate Python environment:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab sports_person_classifier_model.ipynb
```

## Required local assets

The training images are **not included**. The notebook expects:
- `sharapova1.jpg` for the initial demonstration
- `dataset/<person-name>/` with training images
- `dataset/cropped/` for generated crops
- OpenCV face and eye Haar XML files at the paths referenced in the notebook

OpenCV also supplies cascade assets through `cv2.data.haarcascades`; update the notebook paths accordingly. Use images you have permission to process and share.

## Files

`model/saved_model.pkl` and `model/class_dictionary.json` are existing artifacts. Their exact training environment is not recorded. New notebook exports currently go to the root; copy validated exports into `model/` deliberately.

## Limitations

This is a closed-set classification exercise. It does not establish performance on unseen identities or uncontrolled images. Near-duplicate images can inflate a random-split score. Evaluate on independently collected photos, report per-class metrics, and document image sources before quoting performance. A runnable inference API is not yet included.
