# Rate My Professors (RMP) Analysis

<h3>Capstone Project for DS112 (Principles of Data Science) Fall 2024</h3>

This project uses quantitative and qualitative data from 89,893 RMP records to evaluate professor effectiveness through various factors using statistical methods, relationships, and data visualization. The goal is to answer specific questions that include professor ratings and difficulty, biases, and model predictors.

<h4><b>Data Overview</b></h4>
<ul>
  <li>This project contains two datasets:</li>
  1. <b>rmpCapstoneNum.csv:</b> Contains numerical and binary data about professors.
  <ul>
    <li>Columns:</li>
      1. Average Rating<br>
      2. Average Difficulty<br>
      3. Number of Ratings<br>
      4. Pepper? (Boolean - was this professor rated as “hot”?)<br>
      5. Proportion of Students Who Would Take the Class Again<br>
      6. Number of Online Ratings<br>
      7. Male (Boolean – 1 if the Professor is Male)<br>
      8. Female (Boolean – 1 if the Professor is Female)
  </ul>
    2. <b>rmpCapstoneQual.csv:</b> Contains qualitative information about professors.
    <ul>
    <li>Columns:</li>
        1. Major/Field<br>
        2. University<br>
        3. US State (2-Letter Abbreviation)
    </ul>
</ul>

<h4>Questions Answered:</h4>
<p>This project specifically answers 10 primary questions with an Extra Credit section.</p>
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

<h4>Key Steps:</h4>
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
  </ul>
  <li><b>Building Models:</b></li>
  <ul>
    <li>Built Regression Models to predict Average Ratings from Average Difficulty only and then all available factors.</li>
    <li>Built Classification Models to predict Whether a Professor will Receive a "Pepper" from Average Ratings only and then all available factors, using metrics like AUROC and Confusion Matrices.</li>
  </ul>
  <li><b>Data Visualization:</b></li>
  <ul>
    <li>Generated plots including Histograms and Scatter Plots with Regression Lines to illustrate relationships between predictors.</li>
    <li>Classified model performance using metrics like AUROC curves and Confusion Matrices.</li>
  </ul>
</ol>

<h4>Project Deliverables:</h4>
<p>This project includes two deliverable files that contain my answers and codes to the questions.</p>
<ol>
  <li><b></b></li>
</ol>
