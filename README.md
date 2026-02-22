# Vecino Más Cercano Aproximado
Conocer el funcionamiento de ANN 

En un mundo ideal, cuando se busca algo en una **Base de Datos Vectorial**, se quiere encontrar el vector exactamente más cercano **(KNN - k-Nearest Neighbors)**. Sin embargo, el **KNN exacto** requiere comparar tu consulta contra cada uno de los millones (o miles de millones) de vectores almacenados. Esto es computacionalmente imposible para aplicaciones en tiempo real.

**¿Cual es el Problema que se quiere solucionar con ANN?**
- La "Maldición de la Dimensionalidad"

A medida que aumenta la cantidad de datos y dimensiones (los embeddings modernos suelen tener +768 dimensiones), una búsqueda lineal (O(n)) se vuelve demasiado lenta.

**¿Porqué pensar en ANN?**

**ANN** sacrifica un pequeño porcentaje de precisión **(Recall)** a cambio de una mejora masiva en velocidad. En lugar de revisar todo el espacio, **ANN** utiliza estructuras de datos inteligentes para **"saltar"** directamente a la región donde es más probable que vivan los vecinos.

**Algoritmos Principales:**

**HNSW (Hierarchical Navigable Small World):** Crea una estructura de grafo por capas. Es el estándar actual por su altísimo rendimiento.

**IVF (Inverted File Index):** Divide el espacio en clústeres y solo busca en los clústeres más cercanos al query.

**PQ (Product Quantization):** Comprime los vectores para que ocupen menos memoria y las distancias se calculen más rápido.
