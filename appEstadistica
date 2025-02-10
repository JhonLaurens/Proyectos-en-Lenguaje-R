from flask import Flask, request, render_template_string, redirect, url_for
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis, probplot, t
import matplotlib.pyplot as plt
import io
import base64
from math import ceil
import math

app = Flask(__name__)

# Configuración avanzada de estilos
BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
BOOTSTRAP_JS = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"
PLOTLY_JS = "https://cdn.plot.ly/plotly-latest.min.js"
FONT_AWESOME = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AnalyticsPro - Análisis Estadístico Avanzado</title>
    <link href="{{ BOOTSTRAP_CSS }}" rel="stylesheet">
    <link href="{{ FONT_AWESOME }}" rel="stylesheet">
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        .card { margin-bottom: 1.5rem; box-shadow: 0 0.5rem 1rem rgba(0,0,0,.15); }
        .chart-container { height: 400px; }
        .data-table th { background-color: #2c3e50; color: white; }
        .tooltip-icon { color: #2c3e50; cursor: help; }
        .highlight { background-color: #f8f9fa; transition: all 0.3s; }
        .highlight:hover { transform: translateY(-3px); }
        .section-title { border-bottom: 2px solid #2c3e50; padding-bottom: 0.5rem; }
        .stat-badge { font-size: 0.9em; background-color: #e9ecef !important; color: #2c3e50 !important; }
        .download-btn { position: fixed; bottom: 20px; right: 20px; z-index: 1000; }
    </style>
</head>
<body class="bg-light">
    <div class="container py-4">
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h1 class="mb-0"><i class="fas fa-chart-line me-2"></i>AnalyticsPro</h1>
                <p class="lead mb-0">Plataforma de Análisis Estadístico Integral</p>
            </div>
            
            <div class="card-body">
                <form method="post" action="/">
                    <div class="mb-3">
                        <label for="datos" class="form-label"><i class="fas fa-database me-1"></i>Ingrese datos (separados por espacios o comas):</label>
                        <textarea class="form-control" id="datos" name="datos" rows="3" 
                                  placeholder="Ejemplo: 12.5 15.3 14.2 16.8 13.7">{{ datos|default('') }}</textarea>
                    </div>
                    <div class="row g-3 mb-3">
                        <div class="col-md-4">
                            <label class="form-label">Nivel de confianza:</label>
                            <select class="form-select" name="confianza">
                                <option value="0.90" {{ 'selected' if confianza == '0.90' }}>90%</option>
                                <option value="0.95" {{ 'selected' if confianza == '0.95' }}>95%</option>
                                <option value="0.99" {{ 'selected' if confianza == '0.99' }}>99%</option>
                            </select>
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Decimales:</label>
                            <select class="form-select" name="decimales">
                                <option value="2" {{ 'selected' if decimales == '2' }}>2 decimales</option>
                                <option value="3" {{ 'selected' if decimales == '3' }}>3 decimales</option>
                                <option value="4" {{ 'selected' if decimales == '4' }}>4 decimales</option>
                            </select>
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Formato numérico:</label>
                            <select class="form-select" name="formato">
                                <option value="normal" {{ 'selected' if formato == 'normal' }}>Estándar</option>
                                <option value="cientifico" {{ 'selected' if formato == 'cientifico' }}>Científico</option>
                            </select>
                        </div>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">
                        <i class="fas fa-rocket me-2"></i>Analizar Datos
                    </button>
                </form>

                {% if error %}
                    <div class="alert alert-danger mt-3">{{ error }}</div>
                {% endif %}
            </div>
        </div>

        {% if resultados %}
        <!-- Sección de Descarga -->
        <a href="#reporte" class="btn btn-success download-btn" download="reporte.pdf">
            <i class="fas fa-file-pdf me-2"></i>Descargar Reporte
        </a>

        <!-- Panel Resumen -->
        <div class="card highlight">
            <div class="card-header bg-info text-white">
                <h3 class="mb-0"><i class="fas fa-clipboard-list me-2"></i>Resumen Estadístico</h3>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-3 text-center mb-3">
                        <div class="stat-badge p-2 rounded">
                            <div class="text-muted small">Media</div>
                            <div class="h4">{{ resultados.tendencia.media_muestral }}</div>
                        </div>
                    </div>
                    <div class="col-md-3 text-center mb-3">
                        <div class="stat-badge p-2 rounded">
                            <div class="text-muted small">Mediana</div>
                            <div class="h4">{{ resultados.tendencia.mediana }}</div>
                        </div>
                    </div>
                    <div class="col-md-3 text-center mb-3">
                        <div class="stat-badge p-2 rounded">
                            <div class="text-muted small">Desv. Estándar</div>
                            <div class="h4">{{ resultados.dispersion.std_muestral }}</div>
                        </div>
                    </div>
                    <div class="col-md-3 text-center mb-3">
                        <div class="stat-badge p-2 rounded">
                            <div class="text-muted small">Tamaño Muestra</div>
                            <div class="h4">{{ resultados.agrupados.tamano_muestra }}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Gráficos -->
        <div class="row">
            <div class="col-md-6">
                <div class="card highlight">
                    <div class="card-header">
                        <h4 class="mb-0"><i class="fas fa-chart-bar me-2"></i>Distribución de Datos</h4>
                    </div>
                    <div class="card-body">
                        <div id="histogram" class="chart-container"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card highlight">
                    <div class="card-header">
                        <h4 class="mb-0"><i class="fas fa-chart-line me-2"></i>Análisis de Normalidad</h4>
                    </div>
                    <div class="card-body">
                        <div id="qqplot" class="chart-container"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Secciones Detalladas -->
        <div class="card">
            <div class="card-header">
                <h3 class="section-title"><i class="fas fa-calculator me-2"></i>Análisis Detallado</h3>
            </div>
            <div class="card-body">
                <!-- Acordeón de secciones -->
                <div class="accordion" id="analisisAcordeon">
                    <!-- Cada sección como ítem del acordeón -->
                    {% include 'sections/tendencia_central.html' %}
                    {% include 'sections/dispersion.html' %}
                    {% include 'sections/distribucion.html' %}
                    {% include 'sections/inferencia.html' %}
                </div>
            </div>
        </div>
        {% endif %}
    </div>

    <script src="{{ BOOTSTRAP_JS }}"></script>
    <script src="{{ PLOTLY_JS }}"></script>
    <script>
    {% if resultados %}
        // Configuración de gráficos
        const layoutBase = {
            plot_bgcolor: '#f8f9fa',
            paper_bgcolor: '#fff',
            font: {family: 'Arial', size: 12},
            margin: {t: 40, b: 40},
            showlegend: false
        };

        // Histograma
        Plotly.newPlot('histogram', [{
            x: {{ resultados.histograma.data | safe }},
            type: 'histogram',
            marker: {color: '#2c3e50'},
            opacity: 0.7,
            nbinsx: {{ resultados.agrupados.numero_clases }}
        }], {...layoutBase, title: 'Distribución de Frecuencias'});

        // QQ Plot
        const qqData = {{ resultados.qqplot | safe }};
        Plotly.newPlot('qqplot', [{
            x: qqData.theoretical,
            y: qqData.sample,
            mode: 'markers',
            marker: {color: '#e74c3c'}
        }, {
            x: qqData.theoretical,
            y: qqData.fit,
            mode: 'lines',
            line: {color: '#2c3e50'}
        }], {...layoutBase, 
            title: 'Gráfico Q-Q', 
            xaxis: {title: 'Cuantiles Teóricos'},
            yaxis: {title: 'Cuantiles Muestrales'}
        });
    {% endif %}
    </script>
</body>
</html>
"""

# Funciones de análisis mejoradas
def calcular_intervalo_confianza_varianza(x, confianza=0.95):
    n = len(x)
    varianza = np.var(x, ddof=1)
    alpha = 1 - confianza
    chi2_lower = chi2.ppf(1 - alpha/2, n-1)
    chi2_upper = chi2.ppf(alpha/2, n-1)
    lower = (n-1)*varianza/chi2_lower
    upper = (n-1)*varianza/chi2_upper
    return (lower, upper)

def analisis_normalidad(x):
    # Test de Shapiro-Wilk
    stat, p = shapiro(x)
    
    # QQ Plot
    plt.figure()
    qq = probplot(x, plot=plt)
    plt.close()
    
    # Extraer datos del QQ plot
    theoretical = qq[0][0].tolist()
    sample = qq[0][1].tolist()
    slope, intercept, r = qq[1]
    fit = [intercept + slope*xi for xi in theoretical]
    
    return {
        'shapiro_stat': stat,
        'shapiro_p': p,
        'qqplot': {
            'theoretical': theoretical,
            'sample': sample,
            'fit': fit,
            'r_squared': r**2
        }
    }

def generar_qqplot_base64(x):
    # Generar imagen del QQ plot
    plt.figure(figsize=(6, 4))
    probplot(x, plot=plt)
    plt.title('Gráfico Q-Q')
    
    # Convertir a base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.read()).decode('utf-8')

# Resto de funciones mejoradas...

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Configuración de formato
        decimales = int(request.form.get('decimales', 2))
        formato = request.form.get('formato', 'normal')
        confianza = float(request.form.get('confianza', 0.95))
        
        # Procesamiento de datos
        datos = request.form.get('datos', '')
        try:
            x = np.array([float(num) for num in re.split(r'[\s,]+', datos) if num.strip()])
            
            # Análisis completo
            resultados = {
                # ... (análisis existentes)
                'inferencia': {
                    'intervalo_varianza': calcular_intervalo_confianza_varianza(x, confianza),
                    'normalidad': analisis_normalidad(x),
                    'prueba_t': realizar_prueba_t(x)
                },
                'histograma': generar_histograma_data(x),
                'qqplot': generar_qqplot_data(x)
            }
            
            return render_template_string(HTML_TEMPLATE, resultados=resultados)
        
        except Exception as e:
            error = f"Error en los datos: {str(e)}"
    
    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    app.run(debug=True)