#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def main(
    in_csv: str = "results_run2/A_row_normalized.csv",
    out_pdf: str = "signature_consequence_distributions.pdf",
):
    # Load signatures × consequences matrix
    A = pd.read_csv(in_csv, index_col=0)

    # Create a multi-page PDF
    with PdfPages(out_pdf) as pdf:
        for sig_name, row in A.iterrows():
            fig, ax = plt.subplots(figsize=(8, 4))

            # Bar chart for this signature
            x = range(len(row.index))
            ax.bar(x, row.values)

            ax.set_title(f"Consequence distribution for {sig_name}")
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)  # since rows are normalized

            ax.set_xticks(x)
            ax.set_xticklabels(row.index, rotation=45, ha="right")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved plots to {out_pdf}")


if __name__ == "__main__":
    main()
