# Guía de la yuca para WSL/Linux

## 1. Resumen del Proyecto 

* **Proyecto:** `p037`
* **Almacén Compartido:** `/lustre/proyectos/p037/`
* **Repo de GitHub:** `git@github.com:Dante-atc/BigDataBowl.git`

### Directorio Compartido

En la raíz de tu `home` en Yuca, ejecuta:

ln -s /lustre/proyectos/p037 compartido

Este enlace simbólico te llevará al directorio compartido del proyecto (/lustre/proyectos/p037/).

### Pasos para ejecutar una libreta con GPU

Conectarse a Yuca en una terminal:

ssh yuca (si tienes el atajo)

2. Colocarse en la raíz del proyecto (donde están los scripts):

cd ~/BigDataBowl

3. Ejecutar el script para pedir un nodo de cómputo con GPU:

./scripts/gpu_session_start.sh

4. Una vez dentro del nodo, ejecutar el script de Jupyter:

./scripts/jupyter_gpu_start.sh

5. Abrir OTRA terminal aparte (en tu PC local) y hacer el túnel SSH. El script anterior te dará el comando exacto. Se verá similar a esto (reemplaza [NODO] y [USUARIO]):

ssh -N -L 9999:[NODO]:9999 [USUARIO]@yuca.unison.mx

6. Copiar el enlace de la primera terminal (la de Yuca) y agregarlo como kernel en VSCode o tu navegador. Se verá así: http://127.0.0.1:9999/lab?token=... 

7. ¡Listo! Ya puedes correr tu libreta.

## Importante

- NUNCA ejecutes trabajos (scripts de Python, notebooks) en el nodo de login (la terminal donde entras con ssh yuca). El login es solo para editar archivos y enviar trabajos. Usa siempre ./scripts/gpu_session_start.sh para ir a un nodo de cómputo.

- Recuerda que tenemos una cuota de 4 GPUs para el equipo.

- Cuando termines, cierra el túnel SSH (Ctrl+C en la terminal local) y sal de la sesión de Yuca (exit) para liberar los recursos.