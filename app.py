import yfinance as yf
import pandas as pd
import streamlit as st
from prophet import Prophet
from datetime import date
@st.cache_data(ttl=3600) 
def main():
    st.sidebar.title("Predicción de Acciones")

    # Convert the file to an opencv image.
    #st.image("índice.jpg", channels="BGR", width=700)

    #df_list = pd.read_csv('spain_ticker_list.csv')

    # Creacion de los botones de la barra lateral
    #company_name = st.sidebar.selectbox("Chose company", df_list['Company Name'])
    changepoint_range = st.sidebar.number_input('changepoint_range',
                                                 min_value=0.1, max_value=0.99,
                                                 value=0.5, step=0.1)
    changepoint_prior_scale = st.sidebar.number_input('changepoint_prior_scale',
                                             min_value=0.5, max_value=5.0,
                                             value=1.0, step=0.1)
    company_ticker = st.text_input("Ingrese el ticker de la empresa")
    #st.write(company_name)
    #company_ticker = df_list.loc[df_list['Company Name'] == company_name].Ticker.reset_index(drop=True)[0]
    st.write(company_ticker)

    fecha_usuario = st.date_input(
    label="Selecciona una fecha",
    value=date.today(),        # valor por defecto
    format="YYYY-MM-DD"         # formato visible
    )

    # 2. Convertir a string en formato YYYY-MM-DD
    fecha_usuario_str = fecha_usuario.strftime("%Y-%m-%d")

    # 3. Fecha actual
    fecha_actual = date.today()
    fecha_actual_str = fecha_actual.strftime("%Y-%m-%d")

    if st.sidebar.button("Predict", key="predict"):
        #define the ticker symbol
        tickerSymbol = company_ticker
        st.write("Wait 1 minute for the Results")
        st.write("Making predictions...")

        #get data on this ticker
        tickerData = yf.Ticker(tickerSymbol)

        #get the historical prices for this ticker
        tickerDf = tickerData.history(start=fecha_usuario_str, end=fecha_actual_str)

        df = tickerDf['Close']

        df = df.reset_index()
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = df['ds'].dt.tz_localize(None)

        m = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                    changepoint_range=changepoint_range)

        m.add_country_holidays(country_name='US')

        m.fit(df)  # df is a pandas.DataFrame with 'y' and 'ds' columns
        future = m.make_future_dataframe(periods=90)
        predictions = m.predict(future)
        predictions = predictions[predictions['ds'].dt.dayofweek < 5]
        df_pred = predictions[['ds','trend','yhat','yhat_lower','yhat_upper']]
        df_pred['y']=df['y']
        df_pred = df_pred.set_index('ds')

        st.subheader("Prediction")
        st.line_chart(df_pred)

        print(predictions['weekly'])

        fig_components = m.plot_components(predictions)
        st.pyplot(fig_components)

        #see your data
        print(tickerDf.columns)
        st.dataframe(tickerDf)

        #st.dataframe(tickerDf.columns)

        # Verifica si la columna existe

        # Suponiendo que tickerdf es un DataFrame con la columna "Close"
        minimo_anual = tickerDf.loc[tickerDf.groupby(tickerDf.index.year)['Close'].idxmin()]

        # Si deseas obtener solo las fechas del mínimo anual
        #minimo_anual_fecha = minimo_anual['Date']

        st.dataframe(minimo_anual)


if __name__ == '__main__':
    main()