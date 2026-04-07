# Streamlit app for Churn Prediction using ada_boost_churn_model.pkl

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Utility functions
@st.cache_data
def load_data(path='Customer-Churn.csv'):
    df = pd.read_csv(path)
    return df

@st.cache_resource
def build_preprocessor(df):
    # Replicate cleaning steps from the notebook
    telco = df.copy()
    telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')
    telco.dropna(how='any', inplace=True)

    # tenure bin
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['1-12', '13-24', '25-36', '37-48', '49-60', '61-72']
    telco['tenure_bin'] = pd.cut(telco['tenure'], bins=bins, labels=labels, include_lowest=True)

    # Define template X (matching notebook)
    X_template = telco.drop(columns=['customerID', 'Churn', 'tenure'])
    X_template = pd.get_dummies(X_template, drop_first=True)

    # Fit scaler on the template data
    scaler = StandardScaler()
    scaler.fit(X_template)

    # Debug info
    print(f"[DEBUG] build_preprocessor: template_cols={X_template.shape[1]}, template_rows={X_template.shape[0]}")

    return {
        'template_columns': list(X_template.columns),
        'scaler': scaler,
        'tenure_bins': (bins, labels),
        'sample_df': telco
    }


def preprocess_input(user_input, prep):
    # user_input: dict of raw feature values (includes tenure)
    df_in = pd.DataFrame([user_input])

    # compute tenure_bin like notebook
    bins, labels = prep['tenure_bins']
    df_in['tenure_bin'] = pd.cut(df_in['tenure'], bins=bins, labels=labels, include_lowest=True)

    # Drop tenure column (not used in final features)
    df_in = df_in.drop(columns=['tenure'])

    # One hot encoding with same drop_first behavior
    df_in_enc = pd.get_dummies(df_in, drop_first=True)

    # Reindex to template columns, filling missing with 0
    template_cols = prep['template_columns']
    df_in_enc = df_in_enc.reindex(columns=template_cols, fill_value=0)

    # Scale
    scaler = prep['scaler']
    X_scaled = scaler.transform(df_in_enc)

    # Debug info
    nonzero = [(col, int(val)) for col, val in zip(prep['template_columns'], df_in_enc.iloc[0]) if val != 0]
    print(f"[DEBUG] preprocess_input: nonzero_dummies={nonzero[:10]}")
    print(f"[DEBUG] preprocess_input: X_scaled_shape={X_scaled.shape}, first10={X_scaled.flatten()[:10].tolist()}")

    return X_scaled


@st.cache_resource
def load_model(path='ada_boost_churn_model.pkl'):
    print(f"[DEBUG] load_model: loading model from {path}")
    model = joblib.load(path)
    print(f"[DEBUG] load_model: model type={type(model)}")
    return model


def main():
    st.set_page_config(page_title='Churn Prediction', layout='centered')
    st.title('Telecom Customer Churn Prediction ⚡')

    # Load data and preprocessor
    df = load_data()
    prep = build_preprocessor(df)
    model = load_model()

    st.markdown('### Provide customer details to get churn prediction')

    # Prepare input widgets dynamically (exclude tenure_bin from inputs)
    sample = prep['sample_df']

    # We'll ask for tenure (numeric) and other features from sample's columns
    with st.form('input_form'):
        tenure = st.slider('Tenure (months)', min_value=int(sample['tenure'].min()), max_value=int(sample['tenure'].max()), value=12)

        user_input = {'tenure': tenure}

        # For each original column (excluding 'customerID', 'Churn', 'tenure', 'tenure_bin') add an input
        cols_to_ask = [c for c in sample.columns if c not in ['customerID', 'Churn', 'tenure', 'tenure_bin']]


        for col in cols_to_ask:
            if (
                pd.api.types.is_object_dtype(sample[col])
                or pd.api.types.is_categorical_dtype(sample[col])
                or pd.api.types.is_string_dtype(sample[col])
                or pd.api.types.is_bool_dtype(sample[col])
            ):
                opts = sorted(sample[col].dropna().unique().tolist())
                user_input[col] = st.selectbox(col, opts)
            elif pd.api.types.is_integer_dtype(sample[col]):
                minv = int(sample[col].min())
                maxv = int(sample[col].max())
                default = int(sample[col].median())
                user_input[col] = st.number_input(col, value=default, min_value=minv, max_value=maxv, step=1)
            elif pd.api.types.is_numeric_dtype(sample[col]):
                minv = float(sample[col].min())
                maxv = float(sample[col].max())
                default = float(sample[col].median())
                user_input[col] = st.number_input(col, value=default, min_value=minv, max_value=maxv)
            else:
                opts = sorted(sample[col].astype(str).dropna().unique().tolist())
                user_input[col] = st.selectbox(col, opts)   

        submitted = st.form_submit_button('Predict')

    if submitted:
        # Preprocess input and predict
        X_in = preprocess_input(user_input, prep)
        pred_proba = model.predict_proba(X_in)[0][1]
        pred_class = model.predict(X_in)[0]

        # Console debug prints
        print(f"[DEBUG] main: user_input={user_input}")
        print(f"[DEBUG] main: X_in_shape={X_in.shape}, first10={X_in.flatten()[:10].tolist()}")
        print(f"[DEBUG] main: pred_class={int(pred_class)}, pred_proba={float(pred_proba)}")

        st.write('### Prediction Result')
        churn_text = 'Yes' if pred_class == 1 else 'No'
        st.write(f'**Churn:** {churn_text}')
        st.write(f'**Churn probability:** {pred_proba:.2f}')

        # Debug information for troubleshooting
        if st.checkbox('Show debug info'):
            st.write('**Raw input**', user_input)

            # Recompute interim preprocessing steps for display
            bins, labels = prep['tenure_bins']
            df_in = pd.DataFrame([user_input])
            df_in['tenure_bin'] = pd.cut(df_in['tenure'], bins=bins, labels=labels, include_lowest=True)
            df_drop = df_in.drop(columns=['tenure'])

            st.write('**After adding tenure_bin (raw)**')
            st.write(df_in)

            st.write('**One-hot encoded (before reindex)**')
            df_in_enc = pd.get_dummies(df_drop, drop_first=True)
            st.write(df_in_enc)

            st.write('**Reindexed (matches model features)**')
            df_reindexed = df_in_enc.reindex(columns=prep['template_columns'], fill_value=0)
            st.write(df_reindexed)

            st.write('**Scaled features (first 20)**')
            st.write(X_in.flatten()[:20].tolist())

            # Show which dummy features are non-zero
            nonzero = [(col, int(val)) for col, val in zip(prep['template_columns'], df_reindexed.iloc[0]) if val != 0]
            st.write('**Non-zero feature dummies**', nonzero)

        st.success('Prediction complete ✅')

    st.markdown('---')
    st.markdown('**Notes:** This app replicates preprocessing used in the notebook: tenure is binned into `tenure_bin`, categorical variables are one-hot encoded with `drop_first=True`, and features are scaled with `StandardScaler`.')


if __name__ == '__main__':
    main()
