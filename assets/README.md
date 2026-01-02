# Waste Sorting Assistant Assets

This folder contains visual assets for the project documentation.

## Expected Files

Add the following files to this directory:

1. **demo.gif** or **demo.png** - Screenshot or animation of the app in action
2. **training_history.png** - Training accuracy/loss curves from the notebook
3. **confusion_matrix.png** - Confusion matrix showing model performance
4. **sample_predictions.png** - Example predictions on test images

## How to Export from Notebook

In your `GarbageTraining.ipynb`, you can save plots using:

```python
import matplotlib.pyplot as plt

# After creating a plot, save it:
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
```

## For Hugging Face Spaces

When deploying to Hugging Face Spaces, these assets can be referenced in your README.md to create an attractive space page.
