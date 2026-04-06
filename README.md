
# **CNN-based Streamlining of Waste Material Segregation for Improving Waste Management**

## **Objective**
The objective of this project is to implement an effective waste material segregation system using convolutional neural networks (CNNs) that categorizes waste into distinct groups. This process enhances recycling efficiency, minimises environmental pollution, and promotes sustainable waste management practices.

The key goals are:

* Accurately classify waste materials into categories like cardboard, glass, paper, and plastic.
* Improve waste segregation efficiency to support recycling and reduce landfill waste.
* Understand the properties of different waste materials to optimize sorting methods for sustainability.

  ## **Data Understanding**

The Dataset consists of images of some common waste materials.

1. Food Waste
2. Metal
3. Paper
4. Plastic
5. Other
6. Cardboard
7. Glass

**Data Description**

* The dataset consists of multiple folders, each representing a specific class, such as `Cardboard`, `Food_Waste`, and `Metal`.
* Within each folder, there are images of objects that belong to that category.
* However, these items are not further subcategorised. <br> For instance, the `Food_Waste` folder may contain images of items like coffee grounds, teabags, and fruit peels, without explicitly stating that they are actually coffee grounds or teabags.

**Dataset Download Link**
[Download Link](https://woolfaws-prod.s3.ap-south-1.amazonaws.com/sharepoint_zips/Dataset_Waste_Segregation.zip)

## Findings during Dataset Preparation
**Categorical Composition & Taxonomy**

1. The dataset is structured into 7 distinct classes: Cardboard, Food_Waste, Glass, Metal, Paper, Plastic, and Other.

  

2. The "Other" Challenge: The inclusion of an "Other" category is a double-edged sword. While it prevents the model from being forced to misclassify unknown items, it often acts as a "noise" bucket, making it the hardest class for the model to learn a specific mathematical pattern for.


**Significant Class Imbalance**

1. There is a notable disparity in sample volume, with Plastic (2,295 images) outperforming Cardboard (540 images) by a ratio of nearly 4:1.

  

2. The "Majority Bias" Risk: Without intervention, the CNN may develop a "lazy" bias, where it achieves high accuracy simply by predicting "Plastic" most of the time.

  

3. Technical Strategy: To counter this, we utilized Stratified Splitting to ensure the 20% validation set isn't missing the smaller categories (like Cardboard) entirely.


**Dimensional Standardization**

1. All raw *.png* images were downsampled to a uniform $128  \times  128$ pixel resolution.

2.  *Impact*: This significantly reduces the number of trainable parameters, preventing the model from crashing due to memory limits while retaining enough detail to distinguish textures like "crinkled plastic" vs. "smooth glass."


**Mathematical Label Encoding**

1. Labels were translated from directory strings into One-Hot Encoded vectors.

2.  *Why it matters*: This transforms a qualitative category into a quantitative probability target. It allows the final Softmax layer of the CNN to output a multi-class probability distribution where the sum of all predictions equals 1.0.


**Data Partitioning Strategy**

1. A fixed 80/20 split was applied with a random_state for reproducibility.

2.  *The Role of Validation*: This 20% "holdout" set acts as the ultimate truth. By monitoring the Validation Loss during training, we can detect exactly when the model stops learning general features and starts "memorizing" specific training images (Overfitting).


**Identified Complexity: Feature Overlap**

1. The dataset contains high Intra-class Variance (e.g., a "Metal" soda can looks nothing like a "Metal" rusty nail) and Inter-class Similarity (e.g., a clear "Plastic" bottle can look nearly identical to a "Glass" bottle).

2.  *Deep Learning Necessity*: This justifies why we used a deep architecture with Dropout and Batch Normalization—the model needs extra "brainpower" to find the subtle edge-cases that separate these overlapping materials.

## Findings from the Model Results
**Model Architecture & Strategy**

1. The model utilizes a Triple-Block CNN structure. Each block consists of a Conv2D layer (extracting spatial patterns) paired with Batch Normalization.

2. Batch Normalization acts as a stabilizer, ensuring that even if one image is very bright and the next very dark, the "signals" sent to the brain of the model stay within a consistent mathematical range.

3.  *Max-Pooling*: By reducing the $128  \times  128$ grid down at each step, the model learns to care about what the object is (a bottle shape) rather than where exactly it is in the photo.

4.  *Spatial Dropout (0.25)*: Applied in the convolutional blocks to prevent the model from relying too heavily on specific pixels.

5.  *Dense Dropout (0.50)*: Applied in the final decision-making layers. This forces the model to learn multiple redundant ways to identify "Metal" or "Glass," making it much more robust to real-world clutter.


**Training Process & Optimization**

1. The use of the ADAM Optimizer paired with ReduceLROnPlateau created a "Smart Training" environment.

2. We use Early Stopping and Model Checkpoint to ensure that the version of the model saved in best_model.h5 is the peak performer—not just the version from the very last epoch, which might have already started to overfit.


**Performance Metrics & Interpretation**

1. Since the Accuracy, Precision, and Recall are all hovering around 68%, it indicates that the model is balanced in distinguishing between all 7 categories with similar levels of success.

  

2. F1-Score (~68%): This is the "Harmonic Mean" of precision and recall. A score this close to the accuracy suggests that the model handles the trade-off between "False Alarms" (Precision) and "Missed Targets" (Recall) very well.

  

2. Confusion Matrix Insights: Confusion matrix revealed that certain class, such as Plastic and Food_Waste, are classified more accurately that others due to class imbalance.

## Conclusion

1. **Data-related Improvements**
   - Implement data augmentation to address class imbalance
   - Collect more samples for underrepresented classes (especially Cardboard)
   - Include more diverse images within each category

2. **Model Enhancements**
   - Experiment with deeper architectures or pre-trained models
   - Implement class weights to handle imbalanced data
   - Try different optimization strategies and hyperparameters

3. **Practical Applications**
   - Could be integrated into automated waste sorting systems
   - Potential for mobile applications for consumer waste sorting
   - Useful for recycling facilities and waste management education

This demonstrates the potential of CNN-based approaches for waste segregation while highlighting areas for future improvement in both data collection and model architecture.

## Conclusion

### Data Related Enhancements
**The primary bottleneck is currently the Class Imbalance.**
1. Implementing Data Augmentation (rotations, zooms, and horizontal flips) is the most immediate way to "teach" the model that a piece of Cardboard is still cardboard even if it’s upside down or cropped.
2. Focusing future data gathering on underrepresented classes such as Cardboard will help the model.
3. Including images with varied lighting and cluttered backgrounds will improve the model.
### DL Model Improvements
1.  *Transfer Learning*: Instead of training from scratch, you could use a Pre-trained Model (like ResNet50 or MobileNetV2) that has already "seen" millions of general images. You simply "fine-tune" its final layers to recognize your 7 waste categories.

2.  *Class Weighting*: During the next training session, the model can be forced to prioritize learning the rarer classes.

### Potential Applications
1.  *Smart Bin Integration*: Retrofitting bins with low-cost cameras and this model to sort waste at the source.

2.  *Consumer Empowerment*: A mobile app that tells a user exactly which bin an item belongs to, reducing "wish-cycling" (putting non-recyclables in the recycling bin)

3.  *Industrial Efficiency*: Helping sorting facilities process materials faster, lowering costs and increasing the purity of recycled bales.
