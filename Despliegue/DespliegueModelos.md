## Despliegue de modelos de aprendizaje automático


En esta actividad, veremos el despliegue de modelos: el proceso de poner los modelos en uso. 
En particular, vemos cómo empaquetar un modelo dentro de un servicio web, para que otros servicios puedan usarlo. 
También vemos cómo implementar el servicio web en un entorno listo para producción. 

### Modelo de predicción

Para comenzar con la implementación, usamos el modelo que entrenamos anteriormente. 
Primero, en esta sección, revisamos cómo podemos usar el modelo para hacer predicciones y luego vemos cómo guardarlo con Pickle. 

#### Usando el modelo 

Usemos el modelo analizado anteriormente en un cuaderno Jupyter o de Colab para calcular la probabilidad de abandono del siguiente cliente: 

``` 
cliente = {
'customerid': '8879-zkjof',
'gender': 'female',
'seniorcitizen': 0,
'partner': 'no',
'dependents': 'no',
'tenure': 41,
'phoneservice': 'yes',
'multiplelines': 'no',
'internetservice': 'dsl',
'onlinesecurity': 'yes',
'onlinebackup': 'no',
'deviceprotection': 'yes',
'techsupport': 'yes',
'streamingtv': 'yes',
'streamingmovies': 'yes',
'contract': 'one_year',
'paperlessbilling': 'yes',
'paymentmethod': 'bank_transfer_(automatic)',
'monthlycharges': 79.85,
'totalcharges': 3320.75,
}
```

Para predecir si este cliente abandonará, podemos usar la función `predict` que escribimos anteriormente:

```
df = pd.DataFrame([cliente])
y_pred = predict(df, dv, modelo)
y_pred[0]
```

El resultado es una matriz NumPy con un solo elemento: la probabilidad prevista de abandono de este cliente.

**Pregunta:** ¿Cuál es tu resultado?. Analiza tu resultado,

Ahora echemos un vistazo a la función `predict` que escribimos anteriormente para aplicar el modelo a los clientes en el conjunto de validación: 

```
def predict(df, dv, modelo):
    cat = df[categorical + numerical].to_dict(orient='rows')
    X = dv.transform(cat)
    y_pred = modelo.predict_proba(X)[:, 1]
   return y_pred
```

Usarlo para un cliente parece ineficiente e innecesario: creamos un dataframes de un solo cliente solo para convertir este dataframes nuevamente en un diccionario más tarde dentro de predict. 

Llamemos a esta función `predict_cliente`: 

```
def predict_cliente(cliente, dv, modelo):
    X = dv.transform([cliente])
    y_pred = modelo.predict_proba(X)[:, 1]
    return y_pred[0]
```


Usarlo se vuelve más simple: simplemente lo invocamos con nuestro cliente (un diccionario): `predict_cliente(cliente, dv, modelo)`.

**Pregunta** ¿El resultado es el mismo? 

Entrenamos el modelo dentro del Jupyter Notebook que comenzamos en clases anteriores. Este modelo vive allí, y una vez que detengamos el Jupyter Notebook, el modelo entrenado desaparecerá. 
Esto significa que ahora podemos usarlo solo dentro de la computadora portátil y en ningún otro lugar. 

### Uso de Pickle para guardar y cargar el modelo 

Para poder usarlo fuera del notebook, necesitamos guardarlo,y luego, otro proceso puede cargarlo y usarlo..

[Pickle](https://docs.python.org/3/library/pickle.html) es un módulo de serialización/deserialización que ya está integrado en Python: usándolo, podemos guardar un objeto de Python arbitrario 
(con algunas excepciones) en un archivo. Una vez que tenemos un archivo, podemos cargar el modelo desde allí en un proceso diferente. 


### Guardar el modelo

Para guardar el modelo, primero importamos el módulo Pickle y luego usamos la función dump: 

```
import pickle
with open('modelo_trabajo.bin', 'wb') as f_out:
     pickle.dump(modelo, f_in)
``` 

Cuando abrimos un archivo con `open` debemos cerrarlo después de que terminemos de escribir. 
Cuando se usa with sucede automáticamente. Sin `with` nuestro código se vería así: 

```
f_out = open('modelo_trabajo.bin', 'wb')
pickle.dump(model, f_out)
f_out.close()
```
En este caso, sin embargo, guardar solo el modelo no es suficiente: también tenemos un `DictVectorizer` que  entrenamos junto con el modelo. 

Tenemos que salvar a los dos. La forma más sencilla de hacer esto es poner ambos en una tupla al usar pickle:

```
with open('modelo_trabajo.bin', 'wb') as f_out:
     pickle.dump((dv, modelo), f_out)
``` 

##### Cargamos los modelos

Para cargar el modelo, usamos la función `load` de Pickle. Podemos probar esto en el mismo cuaderno.

```
with open('modelo_trabajo.bin', 'rb') as f_in:   
      dv, model = pickle.load(f_in)	         
```

Debido a que guardamos una tupla, la descomprimimos al cargar, por lo que obtenemos tanto el vectorizador como el modelo al mismo tiempo. 

**Ejercicio** Escribe un script de Python simple que cargue el modelo y lo aplique a un cliente. Llama a este archivo `servicio1.py`. Este script  de contener :

- La función predict_single que escribimos anteriormente
- El código para cargar el modelo.
- El código para aplicar el modelo a un cliente. 


Después de guardar el archivo, ejecuta este script en Python.




