import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================
# LANGUAGE PACK
# ============================
lang_pack = {
    "en": {
        "title": "Survey Data Analysis",
        "by_group": "By Group 10",
        "upload": "Upload Google Form responses (CSV/XLSX):",
        "select_ind": "Select Independent Variables",
        "desc_stats": "Descriptive Statistics",
        "hist": "Histogram",
        "bar": "Bar Chart",
        "normality": "Normality Test (Shapiro–Wilk)",
        "datatype": "Detected Data Types",
        "correlation": "Spearman Correlation Analysis",
        "corr_result": "Correlation Results",
        "conclusion": "Conclusion",
        "fsc": "Financial Self-Control Index",
        "info": "Upload your data file to begin."
    },
    "id": {
        "title": "Analisis Data Survei",
        "by_group": "Oleh Kelompok 10",
        "upload": "Unggah data Google Form (CSV/XLSX):",
        "select_ind": "Pilih Variabel Independen",
        "desc_stats": "Analisis Deskriptif",
        "hist": "Histogram",
        "bar": "Diagram Batang",
        "normality": "Uji Normalitas (Shapiro–Wilk)",
        "datatype": "Jenis Tipe Data",
        "correlation": "Analisis Korelasi Spearman",
        "corr_result": "Hasil Korelasi",
        "conclusion": "Kesimpulan",
        "fsc": "Indeks Kontrol Keuangan (FSC Index)",
        "info": "Unggah file untuk memulai."
    }
}

# ============================
# UI LANGUAGE SELECTOR
# ============================
st.sidebar.title("Language / Bahasa")
lang = st.sidebar.radio("Choose:", ["English", "Indonesia"])
lang_key = "en" if lang == "English" else "id"
L = lang_pack[lang_key]

st.title(L["title"])
st.write(f"### {L['by_group']}")

# ============================
# FILE UPLOAD
# ============================
file = st.file_uploader(L["upload"], type=["csv","xlsx"])

if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("Preview Data")
    st.dataframe(df.head())

    # ============================
    # IDENTIFY 6 DEPENDENT ITEMS (YOUR EXACT TEXT)
    # ============================
    dep_items = [
        "I can restrain myself from buying things I don’t need.  \nSaya mampu menahan diri untuk tidak membeli barang yang tidak saya butuhkan.  ",
        "I follow the spending budget that I have set.  \nSaya mengikuti anggaran belanja yang sudah saya tetapkan.   ",
        " I think twice before making an online purchase.  \nSaya mempertimbangkan ulang sebelum melakukan pembelian online. ",
        "I prioritize needs over wants when shopping.\nSaya mengutamakan kebutuhan dibandingkan keinginan saat berbelanja.  ",
        "I rarely feel regret after making a purchase.\nSaya jarang menyesal setelah membeli sesuatu.  ",
        "I feel that I have good control over my monthly expenses.  \nSaya merasa memiliki kontrol yang baik terhadap pengeluaran bulanan saya.  "
    ]

    # Likert mapping
    mapping = {
        "Sangat Tidak Setuju":1, "Tidak Setuju":2, "Netral":3,
        "Setuju":4, "Sangat Setuju":5,
        "Never":1, "Rarely":2, "Sometimes":3, "Often":4, "Very Often":5
    }

    df_num = df.copy()

    for col in df.columns:
        df_num[col] = df_num[col].astype(str).str.strip().map(mapping).fillna(df_num[col])

    # FSC Index
    df_num["FSC_Index"] = df_num[dep_items].mean(axis=1)

    st.subheader(L["fsc"])
    st.write(df_num["FSC_Index"].head())

    # Independent variable selection
    available_ind = [c for c in df.columns if c not in dep_items]
    ind_vars = st.multiselect(L["select_ind"], available_ind)

    # ====================================
    # DATA TYPE DETECTION
    # ====================================
    st.subheader(L["datatype"])
    dtype_info = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str)
    })
    st.dataframe(dtype_info)

    # ====================================
    # DESCRIPTIVE STATISTICS
    # ====================================
    if ind_vars:
        st.subheader(L["desc_stats"])
        st.write(df_num[ind_vars + ["FSC_Index"]].describe())

        # ====================================
        # HISTOGRAM
        # ====================================
        st.subheader(L["hist"])
        for c in ind_vars + ["FSC_Index"]:
            fig, ax = plt.subplots()
            num = pd.to_numeric(df_num[c], errors="coerce")
            ax.hist(num.dropna(), bins=5)
            ax.set_title(f"{c}")
            st.pyplot(fig)

        # ====================================
        # BAR CHART
        # ====================================
        st.subheader(L["bar"])
        for c in ind_vars:
            fig, ax = plt.subplots()
            df[c].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f"{c}")
            st.pyplot(fig)

        # ====================================
        # NORMALITY TEST (Shapiro)
        # ====================================
        st.subheader(L["normality"])
        stat, p = stats.shapiro(df_num["FSC_Index"].dropna())
        st.write(f"Shapiro-Wilk p-value: {p:.4f}")
        if p < 0.05:
            st.write("Not Normally Distributed ❌")
        else:
            st.write("Normally Distributed ✔")

        # ====================================
        # CORRELATION
        # ====================================
        st.subheader(L["correlation"])
        results = []
        for c in ind_vars:
            x = pd.to_numeric(df_num[c], errors="coerce")
            y = df_num["FSC_Index"]
            rho, pval = stats.spearmanr(x, y)
            results.append([c, round(rho,3), round(pval,4)])

        corr_df = pd.DataFrame(results, columns=["Variable","Spearman_rho","p_value"])
        st.subheader(L["corr_result"])
        st.dataframe(corr_df)

        # ====================================
        # CONCLUSION
        # ====================================
        st.subheader(L["conclusion"])
        for i, row in corr_df.iterrows():
            var = row["Variable"]
            r = row["Spearman_rho"]
            p = row["p_value"]

            if p < 0.05:
                if r > 0:
                    st.write(f"✔ {var} memiliki hubungan **positif signifikan** dengan FSC Index.")
                else:
                    st.write(f"✔ {var} memiliki hubungan **negatif signifikan** dengan FSC Index.")
            else:
                st.write(f"• {var} **tidak memiliki hubungan signifikan** dengan FSC Index.")

else:
    st.info(L["info"])
