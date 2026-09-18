const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, ImageRun, TableOfContents,
  PageBreak, LevelFormat, convertInchesToTwip, VerticalAlign, Header, Footer,
  PageNumber, NumberFormat,
} = require("docx");

const R = JSON.parse(fs.readFileSync("/home/claude/ai_model_validation_project/outputs/results.json"));
const CH = "/home/claude/ai_model_validation_project/charts/";

// ---------- helpers ----------
const NAVY = "0C447C";
const DARK = "222222";
const MUTED = "5F5E5A";
const GOOD = "27500A";
const WARN = "633806";
const BAD  = "791F1F";
const LINE = "CCCCCC";

const PAGE_W = 12240, PAGE_H = 15840, MARGIN = 1080; // US Letter, 0.75in margins
const CONTENT_W = PAGE_W - 2 * MARGIN;

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 276 },
    children: [new TextRun({ text, size: 21, color: DARK, ...opts })],
  });
}
function bold(text, opts = {}) {
  return new TextRun({ text, bold: true, size: 21, color: DARK, ...opts });
}
function reg(text, opts = {}) {
  return new TextRun({ text, size: 21, color: DARK, ...opts });
}
function bullet(text) {
  return new Paragraph({
    text, bullet: { level: 0 }, spacing: { after: 100, line: 276 },
    run: { size: 21 },
  });
}
function caption(text) {
  return new Paragraph({
    spacing: { before: 80, after: 260 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, italics: true, size: 18, color: MUTED })],
  });
}
function img(path, w, h) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 0 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width: w, height: h } })],
  });
}
function ratingColor(r) {
  if (r === "Low") return GOOD;
  if (r === "Moderate") return WARN;
  return BAD;
}

// simple 2..N column table with header shading
function simpleTable(headers, rows, widthsPct) {
  const total = CONTENT_W;
  const widths = widthsPct.map((pct) => Math.round(total * pct));
  const mkCell = (text, isHeader, align) =>
    new TableCell({
      width: { size: widths[0], type: WidthType.DXA },
      shading: isHeader ? { fill: "0C447C", type: ShadingType.CLEAR } : undefined,
      verticalAlign: VerticalAlign.CENTER,
      margins: { top: 80, bottom: 80, left: 100, right: 100 },
      children: [new Paragraph({
        alignment: align || AlignmentType.LEFT,
        children: [new TextRun({ text: String(text), bold: isHeader, color: isHeader ? "FFFFFF" : DARK, size: 19 })],
      })],
    });

  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((htext, i) => {
      const cell = mkCell(htext, true, i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER);
      cell.options.width = { size: widths[i], type: WidthType.DXA };
      return cell;
    }),
  });

  const bodyRows = rows.map((r, ri) =>
    new TableRow({
      children: r.map((val, i) => {
        const cell = mkCell(val, false, i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER);
        cell.options.width = { size: widths[i], type: WidthType.DXA };
        if (ri % 2 === 1) cell.options.shading = { fill: "F1EFE8", type: ShadingType.CLEAR };
        return cell;
      }),
    })
  );

  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [headerRow, ...bodyRows],
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color: LINE },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: LINE },
      left: { style: BorderStyle.SINGLE, size: 2, color: LINE },
      right: { style: BorderStyle.SINGLE, size: 2, color: LINE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: LINE },
      insideVertical: { style: BorderStyle.SINGLE, size: 2, color: LINE },
    },
  });
}

// ---------- derived numbers ----------
const perf = R.performance;
const cm = R.confusion_matrix; // [[TN, FP],[FN, TP]]
const fairG = R.fairness.protected_group;
const fairS = R.fairness.gender;
const stab = R.stability;
const dq = R.data_quality;

const dpRatioPct = (fairG.demographic_parity_ratio * 100).toFixed(1);
const dpDiffPts = (fairG.demographic_parity_difference * 100).toFixed(1);

// ---------- overall rating logic (transparent, not hand-waved) ----------
// Performance: AUC 0.696 -> Moderate discrimination power
// Fairness: dp_ratio 0.801 -> technically passes 80% rule but with <2pt margin -> Moderate/elevated
// Stability: PSI 0.075 -> Low
const overallRating = "Approve with Conditions";

// ================= DOCUMENT =================
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 21, color: DARK } },
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", run: { size: 30, bold: true, color: NAVY, font: "Calibri" }, paragraph: { spacing: { before: 360, after: 160 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", run: { size: 24, bold: true, color: NAVY, font: "Calibri" }, paragraph: { spacing: { before: 260, after: 120 } } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", run: { size: 21, bold: true, color: DARK, font: "Calibri" }, paragraph: { spacing: { before: 200, after: 100 } } },
    ],
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 260 } } } }],
    }],
  },
  sections: [
    // ---------- TITLE PAGE ----------
    {
      properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1440, bottom: 1440, left: MARGIN, right: MARGIN } } },
      children: [
        new Paragraph({ spacing: { before: 1600, after: 200 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "MODEL VALIDATION REPORT", bold: true, size: 40, color: NAVY })] }),
        new Paragraph({ spacing: { after: 600 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Consumer Loan Approval Model \u2014 v1.0", size: 26, color: DARK })] }),
        new Paragraph({ spacing: { after: 80 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Independent Model Risk Validation", size: 21, color: MUTED, italics: true })] }),
        new Paragraph({ spacing: { before: 800, after: 40 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Prepared by: Malcolm Riley", size: 21, color: DARK })] }),
        new Paragraph({ spacing: { after: 40 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Role: Model Validation Analyst (Independent from Model Development)", size: 19, color: MUTED })] }),
        new Paragraph({ spacing: { after: 40 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Report Date: September 18, 2026", size: 19, color: MUTED })] }),
        new Paragraph({ spacing: { after: 40 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Classification: Internal \u2014 Portfolio Demonstration (Synthetic Data)", size: 19, color: MUTED })] }),
        new Paragraph({ spacing: { before: 900, after: 40 }, alignment: AlignmentType.CENTER,
          children: [bold("Overall Validation Rating: ", { size: 22 }), new TextRun({ text: overallRating, bold: true, size: 22, color: WARN })] }),
        new Paragraph({ spacing: { before: 600 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Note: Applicant data in this report is synthetically generated for demonstration purposes and does not represent real individuals or real loan decisions. Methodology, metrics, and governance mapping follow standard model risk management practice (e.g., SR 11-7 / OCC 2011-12 style model validation and NIST AI RMF).", italics: true, size: 17, color: MUTED })] }),
      ],
    },

    // ---------- MAIN BODY ----------
    {
      properties: {
        page: {
          size: { width: PAGE_W, height: PAGE_H },
          margin: { top: 1080, bottom: 1080, left: MARGIN, right: MARGIN },
          pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL },
        },
      },
      headers: {
        default: new Header({ children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "Model Validation Report \u2014 Consumer Loan Approval Model v1.0", size: 15, color: MUTED })],
        })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Page ", size: 15, color: MUTED }), new TextRun({ children: [PageNumber.CURRENT], size: 15, color: MUTED })],
        })] }),
      },
      children: [
        new Paragraph({ children: [new TextRun({ text: "Table of Contents", bold: true, size: 26, color: NAVY })], spacing: { after: 240 } }),
        ...[
          "Executive Summary",
          "1. Model Overview & Intended Use",
          "2. Data Description & Quality Assessment",
          "3. Methodology",
          "4. Performance Testing",
          "5. Stability & Population Drift",
          "6. Fairness & Disparate Impact Assessment",
          "7. Limitations & Assumptions",
          "8. Governance Mapping \u2014 NIST AI Risk Management Framework",
          "9. Findings Summary & Recommendations",
          "10. Validation Conclusion & Sign-Off",
        ].map((t) => new Paragraph({
          spacing: { after: 110 },
          children: [new TextRun({ text: t, size: 21, color: DARK })],
        })),
        new Paragraph({ children: [new PageBreak()] }),

        // 0. EXECUTIVE SUMMARY
        h1("Executive Summary"),
        p("This report presents an independent validation of the Consumer Loan Approval Model (v1.0), a gradient-boosted classifier that predicts the probability a consumer loan application should be approved. The validation was performed independently of model development, consistent with standard model risk management practice, and covers five areas: conceptual soundness, data quality, performance testing, stability/drift, and fairness (disparate impact) testing."),
        p(`On a held-out test set of ${perf.n_test.toLocaleString()} applications, the model achieved an accuracy of ${(perf.accuracy*100).toFixed(1)}% and an AUC-ROC of ${perf.roc_auc.toFixed(3)}, indicating moderate discriminatory power. No material data quality issues were identified. Simulated forward-period drift testing found no significant population shift (PSI = ${stab.score_psi.toFixed(3)}, below the 0.10 "no significant change" threshold).`),
        p(`The fairness assessment identified a material finding: applicants in Group B were approved by the model at a ${(fairG.by_group.selection_rate["Group A"]*100).toFixed(1)}% vs. ${(fairG.by_group.selection_rate["Group B"]*100).toFixed(1)}% rate relative to Group A \u2014 a demographic parity ratio of ${fairG.demographic_parity_ratio.toFixed(3)} (${dpRatioPct}%). While this technically clears the commonly used 80% four-fifths screening threshold, it does so with a margin of only ${(fairG.demographic_parity_ratio-0.8).toFixed(3)}, and true positive rates also diverge meaningfully between groups (Group A: ${(fairG.by_group.true_positive_rate["Group A"]*100).toFixed(1)}%, Group B: ${(fairG.by_group.true_positive_rate["Group B"]*100).toFixed(1)}%). Investigation traced this gap to the zip_tier feature, which is correlated with Group membership and independently contributes 5.3% of model importance \u2014 a textbook proxy-discrimination pattern. No comparable material gap was found by gender (parity ratio ${fairS.demographic_parity_ratio.toFixed(3)}).`),
        p(`Overall Validation Rating: ${overallRating}. The model is approved for continued use subject to the remediation and monitoring conditions in Section 9.`),

        // 1. MODEL OVERVIEW
        h1("1. Model Overview & Intended Use"),
        h2("1.1 Purpose"),
        p("The model estimates the probability that a consumer loan application will be approved, based on applicant financial and credit attributes. It is intended to support (not fully automate) underwriting decisions, with outputs feeding a decision workflow that retains human review for declined and borderline applications."),
        h2("1.2 Model Type & Features"),
        p("Algorithm: Gradient Boosted Trees (scikit-learn GradientBoostingClassifier; 150 estimators, max depth 3, learning rate 0.08)."),
        p("Training population: 4,500 applications. Validation (holdout) population: 1,500 applications, stratified by outcome."),
        p("Input features (9): credit_score, annual_income, debt_to_income, employment_length_years, age, num_open_accounts, delinquencies_2yr, loan_amount_requested, zip_tier."),
        p("Protected attributes excluded from model features: gender and protected_group (demographic classification) are excluded from training, consistent with fair-lending practice (ECOA / Regulation B), and are used only downstream by the independent validation team for fairness testing \u2014 mirroring how a compliance-held demographic file is used in practice."),

        // 2. DATA QUALITY
        h1("2. Data Description & Quality Assessment"),
        p(`The underlying dataset contains ${dq.n_rows.toLocaleString()} applications and ${dq.n_features} model features. Data quality checks below were run prior to model training.`),
        simpleTable(
          ["Check", "Result", "Assessment"],
          [
            ["Missing values", "employment_length_years: 1.2% missing; all other features: 0%", "Acceptable \u2014 imputed via median"],
            ["Duplicate applicant IDs", `${dq.duplicate_applicant_ids}`, "None found"],
            ["Class balance (approved / declined)", `${(dq.class_balance["1"]*100).toFixed(1)}% / ${(dq.class_balance["0"]*100).toFixed(1)}%`, "Reasonably balanced \u2014 no resampling required"],
            ["Credit score range", `${dq.credit_score_range[0]} \u2013 ${dq.credit_score_range[1]}`, "Within valid FICO-style range"],
            ["Income outliers (>3 std. dev.)", `${dq.outliers_income_gt_3std} of ${dq.n_rows}`, "Immaterial \u2014 retained, no evidence of data entry error"],
          ],
          [0.30, 0.38, 0.32]
        ),
        p(""),

        // 3. METHODOLOGY
        h1("3. Methodology"),
        p("Validation followed a five-part test plan, aligned to standard model risk management practice and mapped to the NIST AI Risk Management Framework in Section 8:"),
        bullet("Performance testing: discrimination (AUC, precision/recall/F1) and calibration on a held-out test set never seen during training."),
        bullet("Data quality review: completeness, duplication, class balance, and outlier scan."),
        bullet("Stability testing: Population Stability Index (PSI) comparing the validation sample against a simulated forward-period population with shifted credit and income distributions."),
        bullet("Fairness / disparate impact testing: demographic parity, equalized odds, and the four-fifths (80%) rule across protected_group and gender, using the Fairlearn library."),
        bullet("Conceptual soundness review: feature importance review to confirm the model's top drivers are economically sensible and to screen for reliance on proxy variables."),

        // 4. PERFORMANCE
        h1("4. Performance Testing"),
        simpleTable(
          ["Metric", "Value", "Interpretation"],
          [
            ["Accuracy", `${(perf.accuracy*100).toFixed(1)}%`, "Correct decisions overall"],
            ["Precision", `${(perf.precision*100).toFixed(1)}%`, "Of approvals, share that were correct"],
            ["Recall (sensitivity)", `${(perf.recall*100).toFixed(1)}%`, "Of true-approvable applicants, share identified"],
            ["F1 score", `${(perf.f1*100).toFixed(1)}%`, "Balance of precision and recall"],
            ["AUC-ROC", `${perf.roc_auc.toFixed(3)}`, "Moderate discriminatory power (0.5 = random, 1.0 = perfect)"],
            ["Brier score", `${perf.brier_score.toFixed(3)}`, "Lower is better-calibrated; acceptable range"],
          ],
          [0.34, 0.20, 0.46]
        ),
        img(CH + "roc_curve.png", 300, 300),
        caption("Figure 1. ROC curve, holdout test set (n=" + perf.n_test + ")."),
        img(CH + "confusion_matrix.png", 300, 270),
        caption("Figure 2. Confusion matrix, holdout test set."),
        img(CH + "calibration.png", 300, 300),
        caption("Figure 3. Calibration curve \u2014 predicted vs. observed approval rate across 10 quantile bins."),
        p("Finding: The model shows moderate, usable discriminatory power (AUC 0.696). Calibration is reasonable across most probability bins, with mild overconfidence at the extremes \u2014 a common and acceptable pattern for boosted-tree models. No performance-based finding requires remediation."),
        img(CH + "feature_importance.png", 340, 230),
        caption("Figure 4. Feature importance (Gini-based)."),
        p("Credit score is the dominant driver (38.4% importance), followed by annual income and debt-to-income \u2014 all economically sensible for a credit decision. zip_tier contributes 5.3% importance; while modest, this is the feature investigated further in Section 6 given its correlation with protected_group."),

        // 5. STABILITY
        h1("5. Stability & Population Drift"),
        p(`To test how the model would behave against a future applicant population, a forward period was simulated with a modest downward shift in credit score and income, and an upward shift in debt-to-income (representative of a mild economic softening scenario). The resulting model-score PSI was ${stab.score_psi.toFixed(3)}.`),
        simpleTable(
          ["PSI range", "Interpretation", "This model"],
          [
            ["< 0.10", "No significant population change", "\u2713 Score PSI = " + stab.score_psi.toFixed(3)],
            ["0.10 \u2013 0.25", "Moderate shift \u2014 monitor", ""],
            ["> 0.25", "Significant shift \u2014 investigate / consider retraining", ""],
          ],
          [0.24, 0.48, 0.28]
        ),
        img(CH + "psi.png", 340, 227),
        caption("Figure 5. Feature-level PSI, simulated forward period vs. validation sample."),
        p(`Finding: No feature exceeded the 0.10 caution threshold in this simulated scenario (highest: debt_to_income at ${stab.feature_psi.debt_to_income.toFixed(3)}). This is a simulated, moderate stress scenario only \u2014 it does not guarantee stability under a real, larger economic shift. Recommend quarterly PSI monitoring in production (Section 9).`),

        // 6. FAIRNESS
        h1("6. Fairness & Disparate Impact Assessment"),
        p("Fairness metrics were computed by joining model predictions on the holdout set back to a compliance-held demographic file (protected_group, gender) that was never exposed to the model during training. This mirrors standard fair-lending validation practice."),
        h2("6.1 Protected Group"),
        simpleTable(
          ["Metric", "Group A", "Group B", "Gap / Ratio"],
          [
            ["Model approval (selection) rate", `${(fairG.by_group.selection_rate["Group A"]*100).toFixed(1)}%`, `${(fairG.by_group.selection_rate["Group B"]*100).toFixed(1)}%`, `Ratio: ${fairG.demographic_parity_ratio.toFixed(3)}`],
            ["True positive rate", `${(fairG.by_group.true_positive_rate["Group A"]*100).toFixed(1)}%`, `${(fairG.by_group.true_positive_rate["Group B"]*100).toFixed(1)}%`, `Diff: ${(fairG.equalized_odds_difference*100).toFixed(1)} pts (equalized odds)`],
            ["False positive rate", `${(fairG.by_group.false_positive_rate["Group A"]*100).toFixed(1)}%`, `${(fairG.by_group.false_positive_rate["Group B"]*100).toFixed(1)}%`, ""],
            ["Four-fifths (80%) rule", "", "", fairG.four_fifths_pass ? "Passes (marginally)" : "Fails"],
          ],
          [0.34, 0.20, 0.20, 0.26]
        ),
        img(CH + "fairness_protected_group.png", 300, 210),
        caption("Figure 6. Model approval rate by protected group, vs. four-fifths threshold."),
        p(`Finding \u2014 MATERIAL: Group B applicants are approved at ${dpRatioPct}% of Group A's rate. This technically clears the 80% four-fifths screening threshold but with only a ${((fairG.demographic_parity_ratio-0.8)*100).toFixed(1)}-point margin, and the gap is corroborated by a ${(fairG.equalized_odds_difference*100).toFixed(1)}-point equalized-odds difference (true and false positive rates both diverge by group). Group membership itself is not a model input; the mechanism was traced to zip_tier, which is disproportionately assigned to Group B applicants and independently penalizes approval probability \u2014 a proxy-discrimination pattern. This finding should be treated as material regardless of the nominal four-fifths pass, because the ratio sits inside the margin of estimation error for a sample this size and the underlying mechanism (a correlated proxy feature) is a known fair-lending red flag.`),
        h2("6.2 Gender"),
        simpleTable(
          ["Metric", "Female", "Male", "Gap / Ratio"],
          [
            ["Model approval (selection) rate", `${(fairS.by_group.selection_rate["Female"]*100).toFixed(1)}%`, `${(fairS.by_group.selection_rate["Male"]*100).toFixed(1)}%`, `Ratio: ${fairS.demographic_parity_ratio.toFixed(3)}`],
            ["True positive rate", `${(fairS.by_group.true_positive_rate["Female"]*100).toFixed(1)}%`, `${(fairS.by_group.true_positive_rate["Male"]*100).toFixed(1)}%`, `Diff: ${(fairS.equalized_odds_difference*100).toFixed(1)} pts (equalized odds)`],
            ["Four-fifths (80%) rule", "", "", fairS.four_fifths_pass ? "Passes comfortably" : "Fails"],
          ],
          [0.34, 0.20, 0.20, 0.26]
        ),
        img(CH + "fairness_gender.png", 300, 210),
        caption("Figure 7. Model approval rate by gender, vs. four-fifths threshold."),
        p(`Finding: No material disparity was found by gender (parity ratio ${fairS.demographic_parity_ratio.toFixed(3)}, well clear of the 80% threshold). No remediation required on this dimension.`),

        // 7. LIMITATIONS
        h1("7. Limitations & Assumptions"),
        bullet("Data is synthetically generated for this demonstration; real applicant data would require additional PII handling, consent, and regulatory review (e.g., Reg B, FCRA) not modeled here."),
        bullet("The forward-drift scenario in Section 5 is a single simulated stress scenario, not a guarantee of stability under all future conditions."),
        bullet("Fairness testing covered two protected dimensions (protected_group, gender); a production validation should also test age, and intersectional combinations (e.g., Group B x Female)."),
        bullet("Sample size (1,500 holdout) is adequate for headline metrics but limits precision on subgroup-level rates; wider confidence intervals apply to the smaller Group B subgroup."),

        // 8. GOVERNANCE MAPPING
        h1("8. Governance Mapping \u2014 NIST AI Risk Management Framework"),
        p("This validation's activities are mapped below to the four core functions of the NIST AI RMF (Govern, Map, Measure, Manage), for audit traceability."),
        simpleTable(
          ["NIST AI RMF Function", "Activity in this Report", "Status"],
          [
            ["Govern", "Independent validation performed separately from model development; documented sign-off and rating (this report).", "Complete"],
            ["Map", "Intended use, in-scope population, and excluded protected attributes documented (Section 1).", "Complete"],
            ["Measure", "Performance, stability/PSI, and fairness metrics computed against defined thresholds (Sections 4\u20136).", "Complete"],
            ["Manage", "Remediation and monitoring plan defined for the proxy-bias finding (Section 9).", "Open \u2014 tracked"],
          ],
          [0.24, 0.52, 0.24]
        ),

        // 9. FINDINGS & RECOMMENDATIONS
        h1("9. Findings Summary & Recommendations"),
        simpleTable(
          ["#", "Finding", "Severity", "Recommendation", "Owner / Due"],
          [
            ["1", "zip_tier acts as a proxy for protected_group, producing a borderline four-fifths ratio (0.801) and a 9.0-point selection rate gap.", "Material", "Remove or re-engineer zip_tier (e.g., replace with a feature less correlated with protected class); retest fairness metrics before next production release.", "Model Owner \u2014 next release cycle"],
            ["2", "No comparable disparity found by gender.", "Informational", "No action required; continue standard monitoring.", "\u2014"],
            ["3", "Model AUC (0.696) is moderate; some overconfidence at probability extremes.", "Low", "Acceptable for current use; revisit if a higher-performing candidate model is proposed.", "Model Owner \u2014 next model review"],
            ["4", "No significant drift found in simulated stress test, but real-world drift may exceed this scenario.", "Low", "Implement quarterly PSI monitoring in production with a 0.10 / 0.25 alert threshold.", "Model Owner \u2014 ongoing"],
          ],
          [0.06, 0.34, 0.13, 0.33, 0.14]
        ),

        // 10. SIGN OFF
        h1("10. Validation Conclusion & Sign-Off"),
        p(`Overall Validation Rating: ${overallRating}.`),
        p("The model demonstrates acceptable performance and stability for its intended use. However, the proxy-discrimination finding in Section 6.1 is material and must be remediated on the timeline in Section 9 before the model's next production release; continued use in the interim should include enhanced monitoring of approval-rate parity by protected group."),
        new Paragraph({ spacing: { before: 500 }, children: [bold("Prepared by: "), reg("Malcolm Riley, Model Validation Analyst")] }),
        new Paragraph({ spacing: { after: 40 }, children: [bold("Independent of: "), reg("Model Development / Business Unit")] }),
        new Paragraph({ spacing: { after: 40 }, children: [bold("Date: "), reg("September 18, 2026")] }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("/home/claude/ai_model_validation_project/outputs/Model_Validation_Report.docx", buf);
  console.log("written");
});
