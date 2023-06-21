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

- La función `predict_cliente` que escribimos anteriormente
- El código para cargar el modelo.
- El código para aplicar el modelo a un cliente. 


Después de guardar el archivo, ejecuta este script en Python.

### Servicio web

Uno de los frameworks más populares para crear servicios web en Python es [Flask](https://flask.palletsprojects.com/en/2.3.x/), que trataremos a continuación. 

#### Flask 

Una forma de implementar un servicio web en Python es usar Flask. Es bastante liviano, requiere poco código para comenzar y oculta la mayor parte de la complejidad de manejar solicitudes y respuestas HTTP.  

Antes de poner nuestro modelo dentro de un servicio web, cubramos los conceptos básicos del uso de Flask. Para eso, crearemos una función simple y la haremos disponible como un servicio web. Después de cubrir los conceptos básicos, nos encargaremos del modelo. 

**Ejercicio:** Escribe un script llamado `flask_test.py`  realizando los siguientes pasos:

1. Escribe una función de Python simple llamada `hola`: 

```
def hola():
     return 'Hola C8280'
```

2. Importa flask para poder usarlo: 

```
from flask import Flask
```

3. Creamos una aplicación Flask, el objeto central para registrar funciones que deben exponerse en el servicio web. 

4. Especifica cómo llegar a la función asignándole una dirección o una ruta en términos de Flask. 

5. Escribe código y ejecuta el script creado.

#### Usando flask en el modelo 

Para calificar a un cliente, el modelo necesita obtener las características, lo que significa que necesitamos una forma de transferir algunos datos de un servicio (el servicio de campaña) a otro (el servicio de abandono). 

Como formato de intercambio de datos, los servicios web suelen utilizar JSON (notación de objetos Javascript). Es similar a la forma en que definimos los diccionarios en Python: 

```
{
"customerid": "8879-zkjof",
"gender": "female",
"seniorcitizen": 0,
"partner": "no",
"dependents": "no",
...
}
```

Para enviar datos, usamos solicitudes POST, no GET: las solicitudes POST pueden incluir los datos en la solicitud, mientras que GET no. Por lo tanto, para que el servicio de campaña pueda obtener predicciones del servicio de abandono, debemos crear una ruta /predict que acepte solicitudes POST. 

El servicio de abandono analizará los datos JSON sobre un cliente y también responderá en JSON. 

**Ejercicio:**

Modifica el archivo `servicio1.py` llamandolo `servicio2.py` de la siguiente manera:  

1. Primero, agregamos algunas importaciones más en la parte superior del archivo: 

```
from flask import Flask, request, jsonify
```

2. A continuación, crea la aplicación Flask. Llamémoslo `Abandono`: 

```
app = Flask('Abandono')
```

3. Ahora necesitamos crear una función con las siguientes características:
    - Que otiene los datos del cliente en una solicitud
    - Qué invoca `predict_cliente` para puntuar al cliente
    - Responde con la probabilidad de abandono en JSON 

4. Ejecuta aplicación Flask.  

**Pregunta:** ¿Qué mensaje aparece después de que se ejecuta ?.  

Probar este código es un poco más difícil que antes: esta vez, necesitamos usar solicitudes POST e incluir al cliente que queremos calificar en el cuerpo de la solicitud. 

La forma más sencilla de hacer esto es usar la librería [requests](https://pypi.org/project/requests/) en Python. 

**Ejercicio:** Abre el mismo Jupyter Notebook que usamos anteriormente y prueba el servicio web desde allí usando `requests`. 


### Gestión de dependencias 

Para el desarrollo local, Anaconda es una herramienta perfecta: tiene casi todas las librerías que podamos necesitar. Esto, sin embargo, también tiene un inconveniente: ocupa 4 GB cuando se desempaqueta, lo cual es demasiado grande. Para la producción, preferimos tener solo las librerías que realmente necesitamos. 

Además, los diferentes servicios tienen diferentes requisitos. A menudo, estos requisitos entran en conflicto, por lo que no podemos usar el mismo entorno para ejecutar varios servicios al mismo tiempo. 

#### Pipenv

Realizamos la creación de un entorno virtual para cada proyecto: una distribución de Python separada que solo requiere librerías para este proyecto en particular. 

[Pipenv](https://pipenv-es.readthedocs.io/es/latest/) es una herramienta que facilita la gestión de entornos virtuales. 

Después de eso, usamos `pipenv `en lugar de pip para instalar algunas dependencias como: 

```
pipenv install numpy scikit-learn flask
```

Después de finalizar la instalación se crea dos archivos: `Pipenv` y `Pipenv.lock`. 

El archivo Pipenv parece bastante simple (puede cambiar): 

```
[[source]]
   name = "pypi"
       url = "https:/ /pypi.org/simple"
            verify_ssl = true
               [dev-packages]
               [packages]
             numpy = "*"
               scikit-learn = "*"
                   flask = "*"
            [requires]
                 python_version = "3.7"
```

Vemos que este archivo contiene una lista de librerías, así como la versión de Python que usamos. 

El otro archivo, `Pipenv.lock`, contiene las versiones específicas de las librerías que usamos para el proyecto. 

**Pregunta:** Muestras los dos archivos que han resultado en el experimento realizado.

Si alguien necesita trabajar en nuestro proyecto, simplemente necesitas ejecutar el comando de instalación: 

```
pipenv install
```

Este paso primero creará un entorno virtual y luego se instala todas las librerías necesarias de `Pipenv.lock`. 


Una vez instaladas todas las librerías, debemos activar el entorno virtual,  de esta forma, nuestra aplicación utilizará las versiones correctas de las librerías. 

Lo hacemos ejecutando el comando `shell`: 

```
pipenv shell
``` 

Qué nos dice que se está ejecutando en un entorno virtual: `Launching subshell in virtual environment`.

**Ejercicio:** Ejecuta el script para servir: `python servicio2.py`. 

Alternativamente, en lugar de ingresar primero explícitamente al entorno virtual y luego ejecutar el script, realiza estos dos pasos con solo un comando: 

```
pipenv run python servicio2.py
```

¿Qué sucede cuando lo probamos con `requests`?. ¿A aparecido una advertencia en la consola? 


WSGI (web server gateway interface) significa interfaz de puerta de enlace del servidor web, que es una especificación que describe cómo las aplicaciones de Python deben manejar las solicitudes HTTP. 

Sin embargo, abordaremos la advertencia instalando un servidor WSGI de producción. Tenemos múltiples opciones posibles en Python y usaremos [Gunicorn](https://gunicorn.org/). 

Instalamos con `Pipenv` este servidor:  `pipenv install gunicorn`

Este comando instala la librería y la incluye como una dependencia en el proyecto al agregarla a los archivos `Pipenv` y `Pipenv.lock`. 

Ejecutemos la aplicación con Gunicorn: 

```
pipenv run gunicorn --bind 0.0.0.0:9696 :servicio2:app
``` 
**Pregunta:** Si todo va bien, ¿qué mensajes aparecene en la terminal?. 



A diferencia del servidor web incorporado de Flask, Gunicorn está listo para la producción, por lo que no tendrás ningún problema bajo carga cuando comencemos a usarlo. 


**Pregunta** Comprueba con con el mismo código dado anteriormente la respuesta resultante?. 


`Pipenv` es una gran herramienta para administrar dependencias: aísla las librerías requeridas en un entorno separado, lo que nos ayuda a evitar conflictos entre diferentes versiones del mismo paquete. 

### Docker

Hemos visto como gestionar las dependencias de Python con Pipenv. Sin embargo, algunas de las dependencias viven fuera de Python. Lo que es más importante, estas dependencias incluyen el sistema operativo (SO), así como las librerías del sistema.  

[Docker](https://www.docker.com/) resuelve este problema "pero funciona en mi máquina" empaquetando también el sistema operativo y las librerías del sistema en un contenedor Docker, un entorno autónomo que funciona en cualquier lugar donde esté instalado Docker. 

Una vez que el servicio está empaquetado en un contenedor Docker, podemos ejecutarlo en nuestra computadora (independientemente del sistema operativo) o cualquier proveedor de nube pública. 

Veamos cómo usarlo para este proyecto. 

Suponemos que ya tiene Docker instalado. Ver: https://docs.docker.com/engine/install/  


Primero, necesitamos crear una imagen de Docker: la descripción del servicio que incluye todas las configuraciones y dependencias. Docker luego usará la imagen para crear un contenedor. Para hacerlo, necesitamos un [Dockerfile](https://docs.docker.com/engine/reference/builder/), un archivo con instrucciones sobre cómo se debe crear la imagen. 

Construyamos una imagen usando instrucciones de Dockerfile y luego ejecutemos esta imagen en una computadora. 

Vamos a crear un archivo con el nombre Dockerfile y el siguiente contenido (ten en cuenta que el archivo no debe incluir las anotaciones): 

```
FROM python:3.7.5-slim			
ENV PYTHONUNBUFFERED=TRUE           
RUN pip --no-cache-dir install pipenv		
WORKDIR /app				
COPY ["Pipfile", "Pipfile.lock", "./"]		
RUN pipenv install --deploy --system && \ rm -rf /root/.cache   
COPY ["*.py", "modelo_trabajo.bin", "./"] 			

EXPOSE 9696 						
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "servicio2:app" ]    
```

**Investigación:**  Explica línea por línea el contenido del archivo Dockerfile. 

Construyamos la imagen. Lo hacemos ejecutando el comando `build` en Docker:

```
docker build -t modelo-prediccion .
```

El indicador -t nos permite establecer el nombre de la etiqueta para la imagen y el parámetro final, el punto, especifica el directorio con el Dockerfile. 
En este caso, significa que usamos el directorio actual. 

Al final, Docker nos dice que creó con éxito una imagen y la etiquetó como `modelo-prediccion:latest`.  

Estamos listos para usar esta imagen para iniciar un contenedor Docker. 

Usa el comando `run` para eso: 

```
docker run -it -p 9696:9696 modelo-prediccion:latest
```

Especificamos algunos parámetros aquí: 

- El indicador `-it` le dice a Docker que lo ejecutamos desde el terminal y necesitamos ver los resultados. 
- El parámetro `-p` especifica la asignación de puertos. `9696:9696` significa asignar el puerto `9696` en el contenedor al puerto `9696` en la máquina de trabajo. 
- Finalmente, necesitamos el nombre y la etiqueta de la imagen, que en nuestro caso es `modelo-prediccion:latest`. 

Ahora nuestro servicio se ejecuta dentro de un contenedor Docker y podemos conectarnos a él mediante el puerto 9696). Este es el mismo puerto que usamos para la aplicación anteriormente. 

![Docker](https://github.com/kapumota/Cuadernos/blob/main/Despliegue/Docker.png)

**Pregunta:** Comprueba los resultados obtenidos desde el cuaderno entregado.

Docker facilita la ejecución de servicios de forma reproducible. Con Docker, el entorno dentro del contenedor siempre permanece igual. 

Esto significa que si podemos ejecutar el servicio en una computadora portátil, funcionará en cualquier otro lugar. 

Debemos probar que la aplicación función en una computadora, así que ahora veamos cómo ejecutar  en una nube pública e implementarla en AWS. 

#### Despliegue

No ejecutamos servicios de producción en computadoras simples necesitamos servidores especiales para eso. 

[AWS Elastic Beanstalk](https://aws.amazon.com/es/elasticbeanstalk/) es una excelente herramienta para comenzar a servir modelos de aprendizaje automático. 
Es elegible para la capa gratuita y debemos tener cuidado y apagarla tan pronto como ya no lo necesitemos. 

 Las formas más avanzadas de hacerlo involucran sistemas de orquestación de contenedores como AWS ECS o Kubernetes o "sin servidor" con AWS Lambda.

Elastic Beanstalk se ocupa automáticamente de muchas cosas que normalmente necesitamos en producción, incluidas la:

- Implementación del servicio en instancias EC2
- Ampliación: agregando más instancias para manejar la carga durante las horas pico
- Reducción: eliminación de estas instancias cuando la carga desaparece
- Reiniciar el servicio si falla por algún motivo
- Equilibrar la carga entre instancias

También necesitaremos una utilidad especial, la interfaz de línea de comandos (CLI) de Elastic Beanstalk, para usar Elastic Beanstalk. 

La CLI está escrita en Python, por lo que podemos instalarla con `pip`, como cualquier otra herramienta de Python.

Sin embargo, debido a que usamos Pipenv, podemos agregarlo como una dependencia de desarrollo. De esta manera, lo instalaremos solo para el proyecto y no en todo el sistema.

```
pipenv install awsebcli --dev
```

Luego de instalar Elastic Beanstalk, podemos ingresar al entorno virtual del proyecto: `pipenv shell`.

Ahora la CLI debería estar disponible. Vamos a comprobarlo:

```
eb --version
``` 

A continuación, ejecutamos el comando de inicialización:

```
eb init -p docker modelo-abandono
````

Ten en cuenta que usamos `-p` docker: de esta manera, especificamos que este es un proyecto basado en Docker.

Si todo está bien, crea un par de archivos, incluido un archivo `config.yml` en la carpeta `.elasticbeanstalk`.

Ahora podemos probar la aplicación localmente usando el comando `local run`:

```
eb local run --port 9696
```

Esto debería funcionar de la misma manera que en la sección anterior con Docker: primero se una imagen y luego ejecuta el contenedor. 


**Pregunta:** Utiliza el mismo código que antes y comprueba que se obtiene la misma respuesta.

Después de verificar que funciona bien localmente, estamos listos para implementarlo en AWS. Podemos hacer eso con un comando:

```
eb create modelo-abandono-env
```
Este simple comando se encarga de configurar todo lo que necesitamos, desde las instancias EC2 hasta las reglas de escalado automático.

Tardará unos minutos en crear todo. Podemos monitorear el proceso y ver qué  se está haciendo en el terminal. 

**Ejercicio:** La URL (`modelo-abandono-env....us-west 2.elasticbeanstalk.com`) en los registros es importante: así es como llegamos a la aplicación. 
Ahora utiliza usar esta URL para hacer predicciones.

![](https://github.com/kapumota/Cuadernos/blob/main/Despliegue/ElasticBeanStalk.png)

Deberías ver la misma respuesta.


Podemos hacer todo desde la terminal usando la CLI, pero también es posible administrarlo desde la Consola de AWS. Para ello, buscamos allí Elastic Beanstalk y seleccionamos el entorno que acabamos de crear. Para desactivarlo, escoge Terminate deployment en el menú Environment action mediante la Consola de AWS.

Aunque Elastic Beanstalk es apto para la capa gratuita, siempre debemos tenga cuidado y apáguelo en cuanto ya no lo necesitemos.


Alternativamente, usamos la CLI para hacerlo:

```
eb terminate modelo-abandono-env
``` 
Después de unos minutos, la implementación se eliminará de AWS y ya no se podrá acceder a la URL.


