# BDI Panic / Euphoria Model

App Streamlit con un **modelo de pánico/euforia** inspirado en el Panic/Euphoria Model de Citi (Tobias Levkovich), extendido con bonos, FX y curva de tasas. Identidad visual **BDI Consultora Patrimonial Integral**.

## Características

- 9 componentes: momentum, breadth, strength, VIX nivel, VIX term structure, safe haven, junk bond, term spread bonos, DXY
- Salida en **Z-score absoluto** (-3 a +3), no 0–100
- Regímenes con umbrales históricos: PÁNICO ≤ -1, Miedo ≤ -0.5, Neutral, Codicia ≥ +0.5, EUFORIA ≥ +1
- Lookback de 5 años para que los extremos sigan siendo extremos
- Tabla automática de **episodios históricos** PÁNICO/EUFORIA
- Pesos del composite ajustables desde el sidebar
- Sección educativa con fórmulas en LaTeX
- Cache de 30 min

## Probar local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

App pública en [Streamlit Community Cloud](https://share.streamlit.io).

---

BDI Consultora Patrimonial Integral · Material educativo, no es asesoramiento financiero.
