import tensorflow as tf
import tensorflow_hub as hub

def env_info():
    print(f'TF버전: {tf.__version__}')
    print(f'즉시실행모드:{tf.executing_eagerly()}')
    print(f'허브버전:{hub.__version__}')
    print("GPU","사용가능" if tf.config.experimental.list_physical_devices("GPU") else "사용불가능")


def is_number(number):
    try:
        float(number)
        return True
    except ValueError:
        return False