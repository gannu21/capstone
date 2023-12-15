## **Instructions for Using the Trained Model**

### **Best Model**
We have successfully trained and saved the best model using hyperparameter optimization. This model is stored in the saved_models directory and is named best_model.pth.

#### **Importance of Using the Saved Model**

- **Environment Variability:** Since this notebook is designed to run in a Google Colab environment, each session can be different, and there might be variations in seed generations each time you connect. This variability can lead to slight changes in the results upon retraining.

- **Reproducibility:** To ensure consistency and reproducibility of results, it is advised to use the saved best model (best_model.pth) for any further predictions or evaluations. This model represents the optimal configuration found during our hyperparameter tuning process.

#### **Loading and Using the Best Model**

**Load the Model:**

To load the best model, use the following code snippet:

```
checkpoint_path = '/content/saved_models/best_model.pth'
best_model = get_model(device)
best_model.load_state_dict(torch.load(checkpoint_path))
```

**Evaluate or Use the Model:**

Once the model is loaded, you can use it to evaluate on test data or make new predictions. If you wish to evaluate the model's performance on the test dataset, you can use the evaluate_model function:

```
accuracy, true_labels, pred_labels = evaluate_model(best_model, test_loader, device, class_names)
print(f'Test Accuracy: {accuracy}%')
```

##### **Note**
If you need to retrain the model for any reason, keep in mind that due to the stochastic nature of training neural networks, you may not get the exact same results. If reproducibility is critical, it's recommended to use the saved model.

## **Hyperparameter Optimization Summary**

I conducted extensive hyperparameter optimization to identify the best settings for our deep learning model. Due to the complexity of the implementation and the computational limitations of Google Colab, it was not feasible to showcase the entire process in a single notebook. The optimization involved over 25 iterations, spanning a range of learning rates, batch sizes, and epoch numbers.

### **Parameters Tested**
The following hyperparameters were explored in our search:

- Learning Rate: Learning rate is crucial in controlling how much to update the model in response to the estimated error each time the model weights are updated. We tested with multiple values: 0.01, 0.001, and 0.0001.
- Batch Size: Batch size determines the number of samples that will be propagated through the network before updating model parameters. We experimented with sizes of 4, 8, 16, 32, and 64.
- Epochs: This parameter defines the number times that the learning algorithm will work through the entire training dataset. We tried different settings: 5, 10, 12, and 15 epochs.

### **Optimal Parameters Found**
After numerous iterations, the following parameters were found to yield the best performance for our model:

- Learning Rate: 0.0001
- Batch Size: 8
- Epochs: 10
These parameters were determined to be the most effective in training our model, achieving a balance between computational efficiency and model accuracy.

### **Challenges in Colab**
The Google Colab environment, while beneficial for its accessibility and provision of GPU resources, presented challenges in conducting this optimization. Each session in Colab can be unique, with potential variations in seed generations and resource availability. This environment's constraints made it impractical to run all iterations in a single, continuous session.