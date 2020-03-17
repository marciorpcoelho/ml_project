import streamlit as st
import pandas as pd
import pickle
from ml_challenge_options import df_before_dummies_path, models_path, df_processed_path


def main():
    df = get_data(df_before_dummies_path)
    df_dummy = get_data(df_processed_path)

    model = get_clf_model(models_path)

    categorical_cols = list(df.select_dtypes(include='object'))
    numerical_cols = list(df.select_dtypes(include='int64').drop('LABEL', axis=1))
    cat_selections, num_selections = [], []

    st.sidebar.title('Parâmetros:')
    for col in categorical_cols:
        cat_selections.append(st.sidebar.selectbox('Por favor escolha um valor para a categoria {}:'.format(col), ['-'] + list(df[col].unique()), index=0))

    for col in numerical_cols:
        num_selections.append(st.sidebar.slider('Por favor escolha um valor para {}'.format(col), float(df[col].min()), float(df[col].max()), value=df[col].mean()))

    if '-' in cat_selections:
        st.error('Por favor escolha todos os parâmetros para a previsão.')
    else:
        prediction_result, prediction_proba = get_prediction(cat_selections, num_selections, categorical_cols, numerical_cols, model, [col for col in list(df_dummy) if col != 'LABEL'])

        st.write('Previsão: {}'.format(prediction_result[0] + 1))  # Adiciono 1 porque um dos processamentos foi reduzir a LABEL a uma classificação binária
        st.write('Probabilidade: {:.2f}'.format(prediction_proba[0][prediction_result[0]]))


@st.cache()
def get_data(file_path):
    df = pd.read_csv(file_path, index_col=0)

    return df


@st.cache
def get_clf_model(model_path):
    # Escolha do modelo Gradient Descent Classificer:
    # Mantém a accuracy alta, enquanto mantém um equilibro de recall entre as duas classes. É o que possui também a RoC mais alta.
    # Em termos de tempo de previsão, é também dos mais rápidos.
    
    with open(model_path + 'gc_model.sav', 'rb') as f:
        clf_model = pickle.load(f)

    return clf_model


def get_prediction(cat_selections, num_selections, categorical_cols, numerical_cols, model, dummy_cols):
    selection = selection_format(cat_selections, num_selections, categorical_cols, numerical_cols, dummy_cols)

    prediction = model.predict(selection)
    prediction_prob = model.predict_proba(selection)

    return prediction, prediction_prob


def selection_format(cat_sels, num_sels, cat_cols, num_cols, dummy_cols):
    selection_cat = pd.DataFrame(columns=cat_cols)
    dummy_df = pd.DataFrame(columns=dummy_cols)

    selection_cat.loc[0, :] = cat_sels
    selection_df = pd.get_dummies(selection_cat[cat_cols])

    dummy_df.loc[0, :] = [1 if x in list(selection_df) else 0 for x in list(dummy_cols)]

    dummy_df.loc[0, num_cols] = num_sels

    return dummy_df


if __name__ == '__main__':
    main()
