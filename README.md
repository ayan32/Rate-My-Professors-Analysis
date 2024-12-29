# Rate My Professors (RMP) Analysis

<h3>Capstone Project for DS112 (Principles of Data Science) Fall 2024</h3>

This project uses quantitative and qualitative data from 89,893 RMP records to evaluate professor effectiveness through various factors using statistical methods, relationships, and data visualization. The goal is to answer specific questions that include professor ratings and difficulty, biases, and model predictors.

<h3><b>Data Overview</b></h3>
<p>This project contains two datasets.</p>
<ol>
  <li><b>rmpCapstoneNum.csv:</b> Contains numerical and binary data about professors.</li>
  <ul>
    <li>Columns:</li>
    <ol type=1>
      <li>Average Rating</li>
      <li>Average Difficulty</li>
      <li>Number of Ratings</li>
      <li>Pepper? (Boolean - was this professor rated as “hot”?)</li>
      <li>Proportion of Students Who Would Take the Class Again</li>
      <li>Number of Online Ratings</li>
      <li>Male (Boolean – 1 if the Professor is Male)</li>
      <li>Female (Boolean – 1 if the Professor is Female)</li>
    </ol>
  </ul>
  <li><b>rmpCapstoneQual.csv:</b> Contains qualitative information about professors.</li>
    <ul>
      <li>Columns:</li>
      <ol type=1>
        <li>Major/Field</li>
        <li>University</li>
        <li>US State (2-Letter Abbreviation)</li>
      </ol>
    </ul>
</ol>

<h3>Questions Answered</h3>
<p>This project answers 10 specific questions with an additional bonus section.</p>
<ol>
  <li><b>Gender Bias:</b> Tests whether male professors receive higher ratings than female professors.</li>
  <li><b>Teaching Experience:</b> Explores the relationship between Teaching Experience (proxied by Number of Ratings) and the Quality of Teaching (Average Rating).</li>
  <li><b>Ratings vs. Difficulty:</b> Explores the relationship between Average Ratings and Average Difficulty.</li>
  <li><b>Online Ratings Comparison:</b> Tests whether professors with more online classes receive higher or lower ratings than professors with less online classes.</li>
  <li><b>Ratings vs. Would Take Again:</b> Explores the relationship between Average Ratings and Proportion that Would Take the Professor Again.</li>
  <li><b>"Hotness" Factor:</b> Tests whether professors who receive a "pepper" receive higher ratings than professors who don't.</li>
  <li><b>Predict Average Ratings:</b> Uses regression to predict Average Ratings from Average Difficulty only.</li>
  <li><b>Predict Average Ratings:</b> Uses regression to predict Average Ratings from all available factors.</li>
  <li><b>Predict Professor "Hotness":</b> Uses classification to predict Whether a Professor will Receive a "Pepper" from Average Ratings only.</li>
  <li><b>Predict Professor "Hotness":</b> Uses classification to predict Whether a Professor will Receive a "Pepper" from all available factors.</li>
</ol>
<p>Additionally, I analyzed whether professors in STEM fields receive higher or lower ratings and difficulty levels than professors not in STEM fields.</p>

<h3>Key Steps</h3>
<ol>
  <li><b>Data Preprocessing:</b></li>
  <ul>
    <li>Loaded and cleaned the dataset by filtering out irrelevant data, handling missing data, and managing outliers.</li>
    <li>Added a Gender column based on the data from Male and Female columns (0 for Male if Male = 1 and Female = 0, 1 for Female if Male = 0 and Female = 1, or 2 for inconclusive if Male = Female).</li>
    <li>Classified majors into STEM vs. non-STEM fields using keyword mapping.</li>
  </ul>
  <li><b>Statistical Analysis:</b></li>
  <ul>
    <li>Performed Mann-Whitney U Test to assess statistical significance between factors involving Ratings and Difficulty Levels.</li>
    <li>Used an alpha level of 0.005 as the threshold for statistical significance to account for potential false positives and correct for multiple comparisons.</li>
    <li>Used Ordinary Least Squares to compare the effect between teaching experience and the quality of teaching.</li>
    <li>Calculated Pearson and Spearman's Correlation Coefficients to measure the strength of relationships.</li>
    <li>Performed bootstrapping for 95% Confidence Intervals and Cohen's d Effect Size to compare the Ratings and Difficulty ranges between STEM and non-STEM majors and the significance of the outcomes.</li>
  </ul>
  <li><b>Building Models:</b></li>
  <ul>
    <li>Built Regression Models to predict Average Ratings from Average Difficulty only and then all available factors.</li>
    <li>Built Classification Models to predict Whether a Professor will Receive a "Pepper" from Average Ratings only and then all available factors, using metrics like AUROC and Confusion Matrices.</li>
    <li>Performed 80/20 train/test splits and 5-fold cross-validations to assess the reliability of the data while also addressing overfitting concerns.</li>
  </ul>
  <li><b>Data Visualization:</b></li>
  <ul>
    <li>Generated plots including Histograms and Scatter Plots with Regression Lines to illustrate relationships between predictors.</li>
    <li>Classified model performance using metrics like AUROC curves and Confusion Matrices.</li>
  </ul>
</ol>

<h3>Project Deliverables</h3>
<p>This project contains two deliverable files.</p>
<ol>
  <li><b>CapstoneProjectReport.pdf:</b> The project report that contains the full answers to the questions including visuals and results along with an introductory paragraph about data preprocessing.</li>
  <li><b>CapstoneProject.py:</b> The Python code file that produced the data analysis and visuals.</li>
</ol>

<h3>Important Notes</h3>
<ul>
  <li><b>RNG Seeding:</b> To ensure reproducibility, the code begins by seeding the random number generator with a unique number. This ensures that the train/test splits and bootstrapping are keyed consistently to this number.</li>
  <li><b>Missing Data:</b> Missing data was handled by removing rows with missing values as this was a small proportion of after the preprocessed data. This was done separately for each question as they each required different columns, and to maximize the available data throughout the project.</li>
  <li><b>Alpha Level:</b> An alpha level of 0.005 was used throughout the analysis to account for potential false positives and multiple comparisons.</li>
</ul>

<h3>Contact</h3>
For any inquires or suggestions, don't hesitate to reach out to <a href="mailto:andrewyan32@gmail.com">Andrew Yan</a>.
