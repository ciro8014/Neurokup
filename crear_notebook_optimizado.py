#!/usr/bin/env python3
"""
Script para crear el notebook optimizado de prueba_c.ipynb
Reduce de 109 celdas a ~50 celdas, elimina redundancias y organiza profesionalmente
"""

import json
import sys

def crear_celda_markdown(texto):
    """Crea una celda de tipo markdown"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": texto if isinstance(texto, list) else [texto]
    }

def crear_celda_codigo(codigo, outputs=[]):
    """Crea una celda de tipo código"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs,
        "source": [codigo] if isinstance(codigo, list) else [codigo]
    }

def crear_notebook_optimizado():
    """Crea el notebook optimizado con estructura profesional"""

    celdas = []

    # ============================================================================
    # TÍTULO PRINCIPAL
    # ============================================================================
    celdas.append(crear_celda_markdown([
        "# 🎯 ANÁLISIS PREDICTIVO DE SUSCRIPCIÓN BANCARIA\n",
        "## Competencia Kaggle - Machine Learning\n",
        "\n",
        "**Objetivo:** Predecir si un cliente se suscribirá a un depósito a término\n",
        "\n",
        "**Dataset:** Datos de campañas de marketing bancario\n",
        "\n",
        "**Autor:** Ciro\n",
        "\n",
        "---"
    ]))

    # ============================================================================
    # SECCIÓN 1: CONFIGURACIÓN INICIAL
    # ============================================================================
    celdas.append(crear_celda_markdown([
        "## 1. CONFIGURACIÓN INICIAL"
    ]))

    # 1.1 Importación de bibliotecas
    celdas.append(crear_celda_markdown([
        "### 1.1 Importación de Bibliotecas"
    ]))

    celdas.append(crear_celda_codigo([
        "# Manipulación de datos\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "# Visualización\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# Machine Learning\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import (\n",
        "    classification_report,\n",
        "    confusion_matrix,\n",
        "    roc_auc_score,\n",
        "    f1_score\n",
        ")\n",
        "\n",
        "# XGBoost y balanceo\n",
        "from xgboost import XGBClassifier\n",
        "from imblearn.over_sampling import SMOTE\n",
        "\n",
        "# Estadísticas\n",
        "from scipy.stats import mstats\n",
        "\n",
        "# Configuración\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "sns.set_palette(\"husl\")\n",
        "%matplotlib inline\n",
        "\n",
        "print(\"✓ Bibliotecas importadas correctamente\")"
    ]))

    # 1.2 Carga de datos
    celdas.append(crear_celda_markdown([
        "### 1.2 Carga de Datos"
    ]))

    celdas.append(crear_celda_codigo([
        "# Cargar datasets\n",
        "df_sample = pd.read_csv('sample_submission.csv', encoding='utf-8')\n",
        "df_train = pd.read_csv('train.csv', encoding='utf-8')\n",
        "df_test_public = pd.read_csv('test_public.csv', encoding='utf-8')\n",
        "df_test_private = pd.read_csv('test_private.csv', encoding='utf-8')\n",
        "\n",
        "print(f\"Train: {df_train.shape}\")\n",
        "print(f\"Test Public: {df_test_public.shape}\")\n",
        "print(f\"Test Private: {df_test_private.shape}\")"
    ]))

    # ============================================================================
    # SECCIÓN 2: ANÁLISIS EXPLORATORIO DE DATOS
    # ============================================================================
    celdas.append(crear_celda_markdown([
        "---\n",
        "## 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)"
    ]))

    # 2.1 Resumen estadístico
    celdas.append(crear_celda_markdown([
        "### 2.1 Resumen Estadístico"
    ]))

    celdas.append(crear_celda_codigo([
        "# Información general\n",
        "print(\"INFORMACIÓN DEL DATASET\\n\")\n",
        "print(f\"Dimensiones: {df_train.shape}\")\n",
        "print(f\"\\nTipos de datos:\")\n",
        "print(df_train.dtypes.value_counts())\n",
        "\n",
        "# Valores faltantes\n",
        "missing = df_train.isnull().sum()\n",
        "if missing.sum() > 0:\n",
        "    print(f\"\\nValores faltantes:\")\n",
        "    print(missing[missing > 0])\n",
        "else:\n",
        "    print(\"\\n✓ No hay valores faltantes\")\n",
        "\n",
        "# Duplicados\n",
        "duplicados = df_train.duplicated().sum()\n",
        "print(f\"\\nRegistros duplicados: {duplicados}\")"
    ]))

    # 2.2 Variable objetivo
    celdas.append(crear_celda_markdown([
        "### 2.2 Análisis de Variable Objetivo"
    ]))

    celdas.append(crear_celda_codigo([
        "# Distribución de la variable objetivo\n",
        "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n",
        "\n",
        "# Conteo\n",
        "df_train['y'].value_counts().plot(kind='bar', ax=ax[0], color=['#ff6b6b', '#4ecdc4'])\n",
        "ax[0].set_title('Distribución de Suscripciones', fontsize=14, fontweight='bold')\n",
        "ax[0].set_xlabel('Suscripción')\n",
        "ax[0].set_ylabel('Frecuencia')\n",
        "ax[0].tick_params(rotation=0)\n",
        "\n",
        "# Porcentaje\n",
        "df_train['y'].value_counts(normalize=True).plot(kind='pie', ax=ax[1], autopct='%1.1f%%',\n",
        "                                               colors=['#ff6b6b', '#4ecdc4'])\n",
        "ax[1].set_title('Proporción de Suscripciones', fontsize=14, fontweight='bold')\n",
        "ax[1].set_ylabel('')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\nDistribución:\")\n",
        "print(df_train['y'].value_counts())\n",
        "print(f\"\\nBalance: {df_train['y'].value_counts(normalize=True) * 100}\")"
    ]))

    # 2.3 Variables numéricas y categóricas
    celdas.append(crear_celda_markdown([
        "### 2.3 Identificación de Variables"
    ]))

    celdas.append(crear_celda_codigo([
        "# Identificar tipos de variables\n",
        "numerical_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()\n",
        "categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()\n",
        "\n",
        "# Remover variable objetivo\n",
        "if 'y' in categorical_cols:\n",
        "    categorical_cols.remove('y')\n",
        "\n",
        "print(f\"Variables Numéricas ({len(numerical_cols)}): {numerical_cols[:10]}...\")\n",
        "print(f\"\\nVariables Categóricas ({len(categorical_cols)}): {categorical_cols[:10]}...\")"
    ]))

    # ============================================================================
    # SECCIÓN 3: INGENIERÍA Y PREPARACIÓN DE DATOS
    # ============================================================================
    celdas.append(crear_celda_markdown([
        "---\n",
        "## 3. INGENIERÍA Y PREPARACIÓN DE DATOS"
    ]))

    # 3.1 Funciones de limpieza
    celdas.append(crear_celda_markdown([
        "### 3.1 Funciones de Limpieza de Datos"
    ]))

    celdas.append(crear_celda_codigo([
        "def parsear_duracion(duracion_str):\n",
        "    \"\"\"\n",
        "    Convierte '3m 12s' a segundos totales\n",
        "    \"\"\"\n",
        "    if pd.isna(duracion_str):\n",
        "        return np.nan\n",
        "    \n",
        "    duracion_str = str(duracion_str).strip()\n",
        "    \n",
        "    # Si ya es un número\n",
        "    if duracion_str.isdigit():\n",
        "        return int(duracion_str)\n",
        "    \n",
        "    # Parsear formato 'Xm Ys'\n",
        "    minutos = 0\n",
        "    segundos = 0\n",
        "    \n",
        "    partes = duracion_str.split()\n",
        "    for parte in partes:\n",
        "        if 'm' in parte:\n",
        "            minutos = int(parte.replace('m', ''))\n",
        "        elif 's' in parte:\n",
        "            segundos = int(parte.replace('s', ''))\n",
        "    \n",
        "    return minutos * 60 + segundos\n",
        "\n",
        "\n",
        "def parsear_p_dias_correcto(valor):\n",
        "    \"\"\"\n",
        "    Maneja 'nunca' en p_dias\n",
        "    \"\"\"\n",
        "    if pd.isna(valor):\n",
        "        return 999\n",
        "    \n",
        "    if isinstance(valor, str):\n",
        "        if valor.lower() == 'nunca':\n",
        "            return 999\n",
        "        try:\n",
        "            return int(valor)\n",
        "        except:\n",
        "            return 999\n",
        "    \n",
        "    return int(valor)\n",
        "\n",
        "\n",
        "def limpieza_completa_final(df, es_train=True):\n",
        "    \"\"\"\n",
        "    Función de limpieza completa y final de datos\n",
        "    \"\"\"\n",
        "    df_clean = df.copy()\n",
        "    \n",
        "    # Eliminar columnas innecesarias\n",
        "    cols_eliminar = ['id', 'DNI', 'nombre', 'apellido']\n",
        "    df_clean = df_clean.drop(columns=cols_eliminar, errors='ignore')\n",
        "    \n",
        "    # Limpieza de trabajo\n",
        "    trabajo_map = {\n",
        "        'Obrero': 'obrero', 'Jubilado': 'jubilado', 'Retirado': 'retirado',\n",
        "        'Administrativo': 'administrativo', 'Estudiante': 'estudiante',\n",
        "        'Emprendedor': 'emprendedor', 'Gestión': 'gestión', 'Autónomo': 'autónomo',\n",
        "        'tecnico': 'técnico', 'experto técnico': 'técnico',\n",
        "        'criado(a)': 'servicio', 'criada(o)': 'servicio', 'servicios': 'servicio',\n",
        "        'retirado': 'jubilado', 'desocupado': 'desempleado', 'autonomo': 'autónomo',\n",
        "        'SIN INFORMACIÓN': 'desconocido', 'DESCONOCIDO': 'desconocido',\n",
        "        'NO SE SABE': 'desconocido', 'NO SE CONOCE': 'desconocido'\n",
        "    }\n",
        "    df_clean['trabajo'] = df_clean['trabajo'].replace(trabajo_map).str.lower()\n",
        "    \n",
        "    # Limpieza de educación\n",
        "    educacion_map = {\n",
        "        'superios': 'superior', 'primari@': 'primaria', 'SECUNDARIA': 'secundaria',\n",
        "        'SIN INFORMACIÓN': 'desconocido', 'DESCONOCIDO': 'desconocido',\n",
        "        'NO SE SABE': 'desconocido', 'NO SE CONOCE': 'desconocido'\n",
        "    }\n",
        "    df_clean['educación'] = df_clean['educación'].replace(educacion_map).str.lower()\n",
        "    \n",
        "    # Limpieza de estado civil\n",
        "    estado_civil_map = {\n",
        "        'divorcio voluntario': 'divorciado', 'divorcio contencioso': 'divorciado',\n",
        "        'casado, con hijos': 'casado', 'casado, sin hijos': 'casado'\n",
        "    }\n",
        "    df_clean['estado_civil'] = df_clean['estado_civil'].replace(estado_civil_map)\n",
        "    \n",
        "    # Convertir variables binarias\n",
        "    df_clean['hipoteca'] = df_clean['hipoteca'].str.lower()\n",
        "    df_clean['riesgo_crediticio'] = (df_clean['riesgo_crediticio'] == 'Sí').astype(int)\n",
        "    df_clean['deuda_personal'] = (df_clean['deuda_personal'] == 'si').astype(int)\n",
        "    df_clean['hipoteca'] = (df_clean['hipoteca'] == 'si').astype(int)\n",
        "    df_clean['incumplimiento'] = (df_clean['incumplimiento'] == 'si').astype(int)\n",
        "    \n",
        "    # Limpieza de contacto\n",
        "    contacto_map = {\n",
        "        'CEL': 'celular', 'CELL': 'celular', 'celular': 'celular', 'Celular ': 'celular',\n",
        "        'teléfono ': 'telefono', 'TEL': 'telefono', 'Teléfono': 'telefono',\n",
        "        'Desconocido': 'desconocido', 'desconocido': 'desconocido'\n",
        "    }\n",
        "    df_clean['contacto'] = df_clean['contacto'].replace(contacto_map)\n",
        "    \n",
        "    # Limpieza de p_resultado\n",
        "    p_resultado_map = {\n",
        "        'SIN INFORMACIÓN': 'desconocido', 'sin informacion': 'desconocido',\n",
        "        'DESCONOCIDO': 'desconocido', 'NO SE SABE': 'desconocido',\n",
        "        'exito': 'éxito', 'EXITO': 'éxito', 'Exitoso': 'éxito',\n",
        "        'fracaso': 'fracaso', 'FRACASO': 'fracaso'\n",
        "    }\n",
        "    df_clean['p_resultado'] = df_clean['p_resultado'].replace(p_resultado_map).str.lower()\n",
        "    \n",
        "    # Procesar fecha\n",
        "    if 'fecha' in df_clean.columns:\n",
        "        df_clean['fecha'] = pd.to_datetime(df_clean['fecha'], errors='coerce')\n",
        "        df_clean['dia'] = df_clean['fecha'].dt.day\n",
        "        df_clean['mes'] = df_clean['fecha'].dt.month\n",
        "        df_clean = df_clean.drop(columns=['fecha'])\n",
        "    \n",
        "    # Parsear duracion\n",
        "    if 'duracion' in df_clean.columns:\n",
        "        df_clean['duracion'] = df_clean['duracion'].apply(parsear_duracion)\n",
        "    \n",
        "    # Parsear p_dias\n",
        "    df_clean['p_dias'] = df_clean['p_dias'].apply(parsear_p_dias_correcto)\n",
        "    \n",
        "    return df_clean\n",
        "\n",
        "print(\"✓ Funciones de limpieza definidas\")"
    ]))

    # 3.2 Aplicación de limpieza
    celdas.append(crear_celda_markdown([
        "### 3.2 Aplicación de Limpieza"
    ]))

    celdas.append(crear_celda_codigo([
        "# Aplicar limpieza a todos los datasets\n",
        "df_train_clean = limpieza_completa_final(df_train, es_train=True)\n",
        "df_test_public_clean = limpieza_completa_final(df_test_public, es_train=False)\n",
        "df_test_private_clean = limpieza_completa_final(df_test_private, es_train=False)\n",
        "\n",
        "print(\"✓ Limpieza aplicada a todos los datasets\")"
    ]))

    # 3.3 Feature Engineering
    celdas.append(crear_celda_markdown([
        "### 3.3 Feature Engineering"
    ]))

    celdas.append(crear_celda_codigo([
        "def feature_engineering_final(df):\n",
        "    \"\"\"\n",
        "    Feature Engineering completo y final\n",
        "    \"\"\"\n",
        "    df_fe = df.copy()\n",
        "    \n",
        "    # Variables de edad\n",
        "    df_fe['edad'] = 2024 - df_fe['año_nacimiento']\n",
        "    df_fe['grupo_edad'] = pd.cut(\n",
        "        df_fe['edad'],\n",
        "        bins=[0, 25, 35, 45, 55, 65, 100],\n",
        "        labels=['muy_joven', 'joven', 'adulto_joven', 'adulto', 'senior', 'anciano']\n",
        "    )\n",
        "    \n",
        "    # Variables financieras\n",
        "    df_fe['fondos_winsorized'] = mstats.winsorize(\n",
        "        df_fe['fondos_promedio_anual'],\n",
        "        limits=[0.05, 0.05]\n",
        "    )\n",
        "    df_fe['nivel_fondos'] = pd.cut(\n",
        "        df_fe['fondos_winsorized'],\n",
        "        bins=[-10000, 0, 500, 1500, 100000],\n",
        "        labels=['negativo', 'bajo', 'medio', 'alto']\n",
        "    )\n",
        "    \n",
        "    # Score de riesgo financiero\n",
        "    df_fe['alto_riesgo'] = (\n",
        "        (df_fe['riesgo_crediticio'] == 1) |\n",
        "        (df_fe['incumplimiento'] == 1)\n",
        "    ).astype(int)\n",
        "    \n",
        "    df_fe['score_mal_financiero'] = (\n",
        "        df_fe['riesgo_crediticio'] +\n",
        "        df_fe['deuda_personal'] +\n",
        "        df_fe['incumplimiento'] +\n",
        "        (df_fe['fondos_promedio_anual'] < 0).astype(int)\n",
        "    )\n",
        "    \n",
        "    # Variables de contacto\n",
        "    df_fe['contactos_previos_log'] = np.log1p(df_fe['contactos_previos'])\n",
        "    df_fe['fue_contactado_antes'] = (df_fe['p_dias'] < 500).astype(int)\n",
        "    df_fe['ratio_dias_contactos'] = np.where(\n",
        "        df_fe['contactos_previos'] > 0,\n",
        "        df_fe['p_dias'] / (df_fe['contactos_previos'] + 1),\n",
        "        0\n",
        "    )\n",
        "    \n",
        "    # Variables de campaña\n",
        "    df_fe['campaña_bin'] = pd.cut(\n",
        "        df_fe['campaña'],\n",
        "        bins=[0, 1, 2, 3, 5, 100],\n",
        "        labels=['1_contacto', '2_contactos', '3_contactos', '4-5_contactos', '6+_contactos']\n",
        "    )\n",
        "    \n",
        "    # Transformaciones de p_dias\n",
        "    df_fe['p_dias_transformado'] = np.where(\n",
        "        df_fe['p_dias'] == 999,\n",
        "        -1,\n",
        "        df_fe['p_dias']\n",
        "    )\n",
        "    df_fe['p_dias_log'] = np.log1p(df_fe['p_dias_transformado'] + 2)\n",
        "    \n",
        "    # Variables temporales\n",
        "    df_fe['trimestre'] = pd.cut(\n",
        "        df_fe['mes'],\n",
        "        bins=[0, 3, 6, 9, 12],\n",
        "        labels=['Q1', 'Q2', 'Q3', 'Q4']\n",
        "    )\n",
        "    df_fe['fin_de_mes'] = (df_fe['dia'] >= 25).astype(int)\n",
        "    \n",
        "    # Duracion (solo si existe)\n",
        "    if 'duracion' in df_fe.columns and df_fe['duracion'].notna().sum() > 0:\n",
        "        df_fe['duracion_log'] = np.log1p(df_fe['duracion'])\n",
        "        df_fe['duracion_bin'] = pd.cut(\n",
        "            df_fe['duracion'],\n",
        "            bins=[0, 120, 300, 600, 5000],\n",
        "            labels=['muy_corta', 'corta', 'media', 'larga']\n",
        "        )\n",
        "    \n",
        "    return df_fe\n",
        "\n",
        "print(\"✓ Función de feature engineering definida\")"
    ]))

    # 3.4 Aplicación de feature engineering
    celdas.append(crear_celda_markdown([
        "### 3.4 Aplicación de Feature Engineering"
    ]))

    celdas.append(crear_celda_codigo([
        "# Aplicar feature engineering\n",
        "df_train_fe = feature_engineering_final(df_train_clean)\n",
        "df_test_public_fe = feature_engineering_final(df_test_public_clean)\n",
        "df_test_private_fe = feature_engineering_final(df_test_private_clean)\n",
        "\n",
        "print(\"✓ Feature engineering aplicado\")\n",
        "print(f\"\\nNuevas dimensiones train: {df_train_fe.shape}\")"
    ]))

    # 3.5 Definición de features
    celdas.append(crear_celda_markdown([
        "### 3.5 Definición de Features para Modelado"
    ]))

    celdas.append(crear_celda_codigo([
        "# Features numéricas\n",
        "features_numericas = [\n",
        "    'edad',\n",
        "    'fondos_winsorized',\n",
        "    'contactos_previos_log',\n",
        "    'p_dias_transformado',\n",
        "    'p_dias_log',\n",
        "    'dia',\n",
        "    'mes',\n",
        "    'ratio_dias_contactos'\n",
        "]\n",
        "\n",
        "# Features binarias\n",
        "features_binarias = [\n",
        "    'riesgo_crediticio',\n",
        "    'deuda_personal',\n",
        "    'incumplimiento',\n",
        "    'hipoteca',\n",
        "    'fue_contactado_antes',\n",
        "    'fin_de_mes',\n",
        "    'alto_riesgo',\n",
        "    'score_mal_financiero'\n",
        "]\n",
        "\n",
        "# Features categóricas\n",
        "features_categoricas = [\n",
        "    'trabajo',\n",
        "    'estado_civil',\n",
        "    'educación',\n",
        "    'contacto',\n",
        "    'p_resultado',\n",
        "    'campaña_bin',\n",
        "    'nivel_fondos',\n",
        "    'grupo_edad',\n",
        "    'trimestre'\n",
        "]\n",
        "\n",
        "# Lista completa de features (SIN duracion para producción)\n",
        "features_modelo = features_numericas + features_binarias + features_categoricas\n",
        "\n",
        "print(f\"Total de features: {len(features_modelo)}\")\n",
        "print(f\"  - Numéricas: {len(features_numericas)}\")\n",
        "print(f\"  - Binarias: {len(features_binarias)}\")\n",
        "print(f\"  - Categóricas: {len(features_categoricas)}\")"
    ]))

    # 3.6 Train-validation split
    celdas.append(crear_celda_markdown([
        "### 3.6 División Train/Validation"
    ]))

    celdas.append(crear_celda_codigo([
        "# Preparar X e y\n",
        "X = df_train_fe[features_modelo].copy()\n",
        "y = (df_train_fe['y'] == 'si').astype(int)\n",
        "\n",
        "# Split estratificado\n",
        "X_train, X_val, y_train, y_val = train_test_split(\n",
        "    X, y, test_size=0.20, random_state=42, stratify=y\n",
        ")\n",
        "\n",
        "print(f\"Train set: {X_train.shape}, balance: {y_train.mean():.2%}\")\n",
        "print(f\"Val set: {X_val.shape}, balance: {y_val.mean():.2%}\")"
    ]))

    # ============================================================================
    # SECCIÓN 4: MODELADO Y EVALUACIÓN
    # ============================================================================
    celdas.append(crear_celda_markdown([
        "---\n",
        "## 4. MODELADO Y EVALUACIÓN"
    ]))

    # 4.1 Preprocessing pipeline
    celdas.append(crear_celda_markdown([
        "### 4.1 Pipeline de Preprocesamiento"
    ]))

    celdas.append(crear_celda_codigo([
        "# Crear preprocessor\n",
        "preprocessor = ColumnTransformer(\n",
        "    transformers=[\n",
        "        ('num', Pipeline(steps=[\n",
        "            ('imputer', SimpleImputer(strategy='mean')),\n",
        "            ('scaler', StandardScaler())\n",
        "        ]), features_numericas),\n",
        "        ('bin', 'passthrough', features_binarias),\n",
        "        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), features_categoricas)\n",
        "    ],\n",
        "    remainder='drop'\n",
        ")\n",
        "\n",
        "print(\"✓ Pipeline de preprocesamiento creado\")"
    ]))

    # 4.2 Modelo baseline
    celdas.append(crear_celda_markdown([
        "### 4.2 Modelo Baseline - Regresión Logística"
    ]))

    celdas.append(crear_celda_codigo([
        "# Modelo baseline\n",
        "modelo_baseline = Pipeline([\n",
        "    ('preprocessor', preprocessor),\n",
        "    ('classifier', LogisticRegression(\n",
        "        max_iter=1000,\n",
        "        class_weight='balanced',\n",
        "        random_state=42\n",
        "    ))\n",
        "])\n",
        "\n",
        "# Entrenar\n",
        "modelo_baseline.fit(X_train, y_train)\n",
        "\n",
        "# Evaluar\n",
        "y_val_proba = modelo_baseline.predict_proba(X_val)[:, 1]\n",
        "y_val_pred = modelo_baseline.predict(X_val)\n",
        "\n",
        "print(\"Baseline - Regresión Logística:\")\n",
        "print(f\"  AUC-ROC: {roc_auc_score(y_val, y_val_proba):.4f}\")\n",
        "print(f\"  F1-Score: {f1_score(y_val, y_val_pred):.4f}\")"
    ]))

    # 4.3 Random Forest
    celdas.append(crear_celda_markdown([
        "### 4.3 Modelo Random Forest"
    ]))

    celdas.append(crear_celda_codigo([
        "# Random Forest\n",
        "modelo_rf = Pipeline([\n",
        "    ('preprocessor', preprocessor),\n",
        "    ('classifier', RandomForestClassifier(\n",
        "        n_estimators=100,\n",
        "        max_depth=10,\n",
        "        class_weight='balanced',\n",
        "        random_state=42,\n",
        "        n_jobs=-1\n",
        "    ))\n",
        "])\n",
        "\n",
        "modelo_rf.fit(X_train, y_train)\n",
        "\n",
        "# Evaluar\n",
        "y_val_proba_rf = modelo_rf.predict_proba(X_val)[:, 1]\n",
        "\n",
        "# Optimizar threshold\n",
        "best_f1 = 0\n",
        "best_threshold = 0.5\n",
        "\n",
        "for threshold in np.arange(0.20, 0.60, 0.05):\n",
        "    y_val_pred_th = (y_val_proba_rf >= threshold).astype(int)\n",
        "    f1 = f1_score(y_val, y_val_pred_th)\n",
        "    if f1 > best_f1:\n",
        "        best_f1 = f1\n",
        "        best_threshold = threshold\n",
        "\n",
        "print(f\"Random Forest:\")\n",
        "print(f\"  AUC-ROC: {roc_auc_score(y_val, y_val_proba_rf):.4f}\")\n",
        "print(f\"  Best F1: {best_f1:.4f} (threshold={best_threshold:.2f})\")"
    ]))

    # 4.4 XGBoost con SMOTE
    celdas.append(crear_celda_markdown([
        "### 4.4 Modelo Final - XGBoost con SMOTE"
    ]))

    celdas.append(crear_celda_codigo([
        "# Preprocesar datos\n",
        "X_train_processed = preprocessor.fit_transform(X_train)\n",
        "X_val_processed = preprocessor.transform(X_val)\n",
        "\n",
        "# Aplicar SMOTE\n",
        "smote = SMOTE(random_state=42, k_neighbors=5)\n",
        "X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)\n",
        "\n",
        "print(f\"Antes SMOTE: {X_train_processed.shape}, balance: {y_train.mean():.2%}\")\n",
        "print(f\"Después SMOTE: {X_train_smote.shape}, balance: {y_train_smote.mean():.2%}\")\n",
        "\n",
        "# Entrenar XGBoost\n",
        "modelo_xgb = XGBClassifier(\n",
        "    n_estimators=200,\n",
        "    max_depth=6,\n",
        "    learning_rate=0.05,\n",
        "    subsample=0.8,\n",
        "    colsample_bytree=0.8,\n",
        "    random_state=42,\n",
        "    n_jobs=-1\n",
        ")\n",
        "\n",
        "modelo_xgb.fit(X_train_smote, y_train_smote)\n",
        "\n",
        "# Evaluar y optimizar threshold\n",
        "y_val_proba_xgb = modelo_xgb.predict_proba(X_val_processed)[:, 1]\n",
        "\n",
        "best_f1_xgb = 0\n",
        "best_threshold_xgb = 0.5\n",
        "\n",
        "for threshold in np.arange(0.20, 0.60, 0.01):\n",
        "    y_val_pred_th = (y_val_proba_xgb >= threshold).astype(int)\n",
        "    f1 = f1_score(y_val, y_val_pred_th)\n",
        "    if f1 > best_f1_xgb:\n",
        "        best_f1_xgb = f1\n",
        "        best_threshold_xgb = threshold\n",
        "\n",
        "# Predicción final\n",
        "y_val_pred_final = (y_val_proba_xgb >= best_threshold_xgb).astype(int)\n",
        "\n",
        "print(f\"\\nXGBoost + SMOTE (MODELO FINAL):\")\n",
        "print(f\"  AUC-ROC: {roc_auc_score(y_val, y_val_proba_xgb):.4f}\")\n",
        "print(f\"  Best F1: {best_f1_xgb:.4f} (threshold={best_threshold_xgb:.2f})\")\n",
        "print(f\"\\nClassification Report:\\n{classification_report(y_val, y_val_pred_final, target_names=['No', 'Sí'])}\")"
    ]))

    # 4.5 Visualización de matriz de confusión
    celdas.append(crear_celda_markdown([
        "### 4.5 Matriz de Confusión - Modelo Final"
    ]))

    celdas.append(crear_celda_codigo([
        "# Matriz de confusión\n",
        "cm = confusion_matrix(y_val, y_val_pred_final)\n",
        "\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)\n",
        "plt.title('Matriz de Confusión - XGBoost + SMOTE', fontsize=14, fontweight='bold')\n",
        "plt.ylabel('Real')\n",
        "plt.xlabel('Predicho')\n",
        "plt.xticks([0.5, 1.5], ['No', 'Sí'])\n",
        "plt.yticks([0.5, 1.5], ['No', 'Sí'])\n",
        "plt.show()"
    ]))

    # ============================================================================
    # SECCIÓN 5: GENERACIÓN DE PREDICCIONES FINALES
    # ============================================================================
    celdas.append(crear_celda_markdown([
        "---\n",
        "## 5. GENERACIÓN DE PREDICCIONES FINALES"
    ]))

    # 5.1 Preparación de test
    celdas.append(crear_celda_markdown([
        "### 5.1 Preparación de Datos de Test"
    ]))

    celdas.append(crear_celda_codigo([
        "# Guardar IDs antes de combinar\n",
        "ids_public = df_test_public['id'].values\n",
        "ids_private = df_test_private['id'].values\n",
        "ids_combined = np.concatenate([ids_public, ids_private])\n",
        "\n",
        "# Combinar test public y private\n",
        "df_test_combined = pd.concat([\n",
        "    df_test_public_fe,\n",
        "    df_test_private_fe\n",
        "], axis=0, ignore_index=True)\n",
        "\n",
        "# Extraer features\n",
        "X_test_combined = df_test_combined[features_modelo].copy()\n",
        "\n",
        "# Aplicar preprocessing\n",
        "X_test_combined_processed = preprocessor.transform(X_test_combined)\n",
        "\n",
        "print(f\"Test combined: {X_test_combined_processed.shape}\")\n",
        "print(f\"IDs guardados: {len(ids_combined)}\")"
    ]))

    # 5.2 Predicciones
    celdas.append(crear_celda_markdown([
        "### 5.2 Predicciones con Modelo Final"
    ]))

    celdas.append(crear_celda_codigo([
        "# Predicciones con XGBoost + SMOTE\n",
        "y_test_proba = modelo_xgb.predict_proba(X_test_combined_processed)[:, 1]\n",
        "y_test_pred = (y_test_proba >= best_threshold_xgb).astype(int)\n",
        "\n",
        "# Convertir a formato 'si'/'no'\n",
        "y_test_labels = ['si' if pred == 1 else 'no' for pred in y_test_pred]\n",
        "\n",
        "print(f\"Predicciones generadas: {len(y_test_labels)}\")\n",
        "print(f\"Distribución:\")\n",
        "print(f\"  'si': {sum(y_test_pred)} ({sum(y_test_pred)/len(y_test_pred)*100:.1f}%)\")\n",
        "print(f\"  'no': {len(y_test_pred) - sum(y_test_pred)} ({(1-sum(y_test_pred)/len(y_test_pred))*100:.1f}%)\")"
    ]))

    # 5.3 Generación de submission
    celdas.append(crear_celda_markdown([
        "### 5.3 Generación de Archivo Submission"
    ]))

    celdas.append(crear_celda_codigo([
        "# Crear DataFrame submission\n",
        "df_submission = pd.DataFrame({\n",
        "    'id': ids_combined,\n",
        "    'y': y_test_labels\n",
        "})\n",
        "\n",
        "# Guardar archivo\n",
        "filename = 'xgboost_smote_threshold040_submission.csv'\n",
        "df_submission.to_csv(filename, index=False)\n",
        "\n",
        "print(f\"✓ Archivo guardado: {filename}\")\n",
        "print(f\"\\nPrimeras filas:\")\n",
        "print(df_submission.head(10))"
    ]))

    # 5.4 Verificación final
    celdas.append(crear_celda_markdown([
        "### 5.4 Checklist de Verificación Final"
    ]))

    celdas.append(crear_celda_codigo([
        "# Checklist de verificación\n",
        "import os\n",
        "\n",
        "checks = [\n",
        "    (\"Archivo existe\", os.path.exists(filename)),\n",
        "    (\"Número de filas correcto\", len(df_submission) == len(df_test_public) + len(df_test_private)),\n",
        "    (\"Columnas correctas\", list(df_submission.columns) == ['id', 'y']),\n",
        "    (\"Valores correctos\", set(df_submission['y'].unique()) == {'si', 'no'}),\n",
        "    (\"Sin valores nulos\", df_submission.isnull().sum().sum() == 0),\n",
        "    (\"IDs únicos\", df_submission['id'].nunique() == len(df_submission))\n",
        "]\n",
        "\n",
        "print(\"VERIFICACIÓN FINAL:\\n\")\n",
        "all_ok = True\n",
        "for check_name, result in checks:\n",
        "    status = \"✓\" if result else \"✗\"\n",
        "    print(f\"  [{status}] {check_name}\")\n",
        "    if not result:\n",
        "        all_ok = False\n",
        "\n",
        "if all_ok:\n",
        "    print(f\"\\n{'='*50}\")\n",
        "    print(\"🎉 LISTO PARA SUBIR A KAGGLE 🎉\")\n",
        "    print(f\"{'='*50}\")\n",
        "else:\n",
        "    print(\"\\n⚠️ Hay errores que corregir\")"
    ]))

    # ============================================================================
    # SECCIÓN 6: CONCLUSIONES
    # ============================================================================
    celdas.append(crear_celda_markdown([
        "---\n",
        "## 6. CONCLUSIONES Y PRÓXIMOS PASOS\n",
        "\n",
        "### Resumen de Resultados\n",
        "- **Modelo Final:** XGBoost con SMOTE\n",
        "- **Threshold Óptimo:** Determinado mediante optimización en validation set\n",
        "- **Archivo Generado:** `xgboost_smote_threshold040_submission.csv`\n",
        "\n",
        "### Aprendizajes Clave\n",
        "1. El balanceo de clases con SMOTE mejoró significativamente el F1-Score\n",
        "2. La optimización del threshold es crítica para maximizar F1\n",
        "3. XGBoost superó a Random Forest y Regresión Logística\n",
        "\n",
        "### Mejoras Futuras\n",
        "- Probar otros algoritmos (LightGBM, CatBoost)\n",
        "- Feature engineering adicional basado en análisis de errores\n",
        "- Ensemble de múltiples modelos\n",
        "- Hyperparameter tuning con GridSearchCV\n",
        "- Validación cruzada estratificada"
    ]))

    # ============================================================================
    # CREAR ESTRUCTURA DEL NOTEBOOK
    # ============================================================================
    notebook = {
        "cells": celdas,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook


def main():
    """Función principal"""
    print("Creando notebook optimizado...")

    notebook = crear_notebook_optimizado()

    # Guardar notebook
    output_path = "/home/user/Neurokup/prueba_c_optimizado.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Notebook optimizado creado exitosamente!")
    print(f"  Ubicación: {output_path}")
    print(f"  Total de celdas: {len(notebook['cells'])}")
    print(f"\nComparación:")
    print(f"  Original: 109 celdas")
    print(f"  Optimizado: {len(notebook['cells'])} celdas")
    print(f"  Reducción: {109 - len(notebook['cells'])} celdas ({(109 - len(notebook['cells']))/109*100:.1f}%)")


if __name__ == "__main__":
    main()
