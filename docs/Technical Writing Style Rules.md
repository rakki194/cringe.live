# Technical Writing Style Rules

This document outlines the technical writing style used for creating detailed guides and documentation on LoRA training processes and related technical subjects. It emphasizes clear, unambiguous instructions that serve both human readers and AI assistants. The style integrates precise formatting, complete technical details, and advanced features such as LaTeX syntax for mathematical expressions and Hugo-specific frontmatter and shortcodes for structured dynamic content.

## General Formatting Conventions

### Fenced Code Blocks

- All code blocks are surrounded by a blank line before and after to ensure proper markdown rendering.
- The language specifier (e.g., "bash", "yaml", "markdown") is always provided to enable accurate syntax highlighting.

### YAML Frontmatter

- Every document starts with a YAML frontmatter block enclosed within triple-dashed lines.
- Mandatory keys include: title, description, summary, weight, bookToC, bookFlatSection, and aliases.

- For English content, aliases include the "/en/docs/" root; for other languages, language-specific prefixes such as "/hu/docs/" or "/ja/docs/" are used to ensure clear content organization across translations.

### Hugo Shortcodes

- Hugo shortcodes are employed throughout the content to embed dynamic elements. These include embedding images with blurhash, collapsible content sections, and linking related posts.
- Shortcodes must follow exact spacing and syntax. For example:

```md
{{< blurhash
  src="https://example.com/image.png"
  blurhash="LsJIIS%fSwV@.Tb0kDaepIjEs;f*"
  width="1528"
  height="1528"
  alt="Detailed description of the image."
  grid="true"
>}}
```

- HTML comments are used to disable specific markdownlint rules (e.g., MD025 and MD033) to avoid false positives on Hugo templating syntax.

```md
<!--markdownlint-disable MD025 MD033 -->
```

## LaTeX Syntax

### Mathematical Expressions and Notation

LaTeX is employed to clearly render mathematical formulas, ensuring that complex equations related to LoRA training are displayed in high quality.

Display math environments use double dollar signs ($$\n...\n$$) to isolate equations on their own lines. For example:

```latex
$$
W = m\frac{V+\Delta V}{||V+\Delta V||_c}
$$
```

This format is ideal for complex transformations and gradient formulas, enhancing readability. Inline mathematical expressions appear within single dollar signs ($ ... $) to seamlessly integrate formulas within paragraphs; for example, short expressions like $||W||_c$ or $v_j = \frac{w_j}{||w_j||_2}$ can be incorporated effortlessly. Care is taken to ensure that characters such as backslashes and underscores are properly escaped so that they render correctly in markdown processors that support LaTeX. In addition, more complex derivations or proofs may be included along with detailed explanatory text, allowing the reader to grasp both the mathematical rigor and its practical application to LoRA training. Moreover, LaTeX formatting is employed to denote key variables, parameters, and operations (e.g., normalization, gradient calculations), which are essential for accurately conveying experimental results and theoretical insights.

## Hugo Features and Their Use

### Frontmatter Specifications

- The YAML frontmatter provides metadata that Hugo uses to index and bundle the content correctly. This includes creating multiple URL aliases ensuring content is discoverable under various paths.
- The uniformity in frontmatter keys and alias patterns guarantees that documentation remains well-structured and easily navigable, crucial for technical topics like LoRA training details.

### Hugo Shortcodes and Dynamic Content


- Hugo shortcodes are used extensively to provide dynamic, reusable content components within the documentation. In addition to blurhash images, shortcodes include those for collapsible sections and related posts linking.

An example of a collapsible section shortcode is:

```md
{{< section details >}}
Detailed explanatory content goes here.
{{< /section >}}
```

This allows embedding of long explanations or code samples that can be expanded or collapsed as required by the reader.

- Dynamic features allow for the integration of live examples, visualizations, and even interactive code snippets which are crucial in demonstrating training workflows and performance metrics.

3.3 Multilingual Content Management

- A strict convention is followed where English content uses "/en/docs/" as the base alias, while other languages (e.g., Hungarian or Japanese) use their language codes ("/hu/docs/", "/ja/docs/").
- This not only ensures proper indexing in Hugo but also maintains a consistent file and URL structure across multilingual documentation.

3.4 Integration of Technical Images and Visual Data

- Images, graphs, and plots (often representing LoRA training results or model performance metrics) are incorporated using Hugo's dynamic image embedding features coupled with blurhash.
- The blurhash shortcode renders a low-detail placeholder until the full image loads, which enhances user experience without sacrificing technical clarity.

4. Structural and Stylistic Guidelines

4.1 Clarity, Detail, and Reproducibility

- Documentation is segmented into logically ordered sections, often starting with an overview, followed by detailed step-by-step instructions, examples, and concluding with technical analyses.
- Every code snippet, command-line instruction, and mathematical derivation is provided in full detail. For example, a full explanation of LoRA weight decomposition is paired with both LaTeX equations and descriptive text.
- 
4.2 Consistency and Technical Precision

- Consistency in technical terminology is paramount. Commands (such as '--v_parameterization' or 'clip_skip') are always mentioned exactly as they are expected to be used.
- The approach ensures that both novices and experts can follow the documentation without ambiguity, fostering a reproducible environment for technical experiments and model training.


// 5. Application for LoRA Training Documentation and Analysis

// 5.1 Conveying LoRA Training Results
// - The documentation frequently integrates LaTeX-rendered formulas to communicate results from experiments (e.g., gradient analysis, weight normalization, etc.).
// - Graphs and plots are described with accompanying Hugo shortcodes to embed interactive or high-quality static visualizations of training performance.

// 5.2 Integration with AI-Assisted Drafting
// - The guidelines ensure that both AI and human contributors adhere strictly to the formatting rules, which helps in maintaining consistency across documents generated or edited in collaborative workflows.
// - Precise instructions for LaTeX, markdown formatting, and Hugo features make automated content generation more reliable and accurate.
