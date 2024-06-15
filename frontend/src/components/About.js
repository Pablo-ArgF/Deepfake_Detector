import {
  Heading,
  Text,
  Flex,
  Image
} from '@chakra-ui/react';


const About = () => {

    return(
        <Flex direction="column" align="center" padding="2em">
            <Heading as="h1" size="5xl" mb={15}>
                Información del proyecto
            </Heading>
            <Flex direction="Row"  placeContent={'center'} borderRadius="md" p={4}>
                <Flex direction="column" align="center" borderRadius="md" p={4}>
                    <Image h="20em" src="./face.jpg" alt="Pablo Argallero Fernández" />
                    <Flex direction="row" w={'100%'} alignItems="flex-end" borderRadius="md"  p={4} >
                        <Image  borderRadius='full' w={'3em'} cursor={'pointer'} src='./github.png' onClick={()=>{window.open('https://github.com/Pablo-ArgF', '_blank')}} alt='Acceso al Github de Pablo-ArgF'/>
                        <Image  borderRadius='full' w={'3em'} cursor={'pointer'} src='./linkedin.png' onClick={()=>{window.open('https://www.linkedin.com/in/pablo-argallero/', '_blank')}} alt='Acceso al LinkedIn de Pablo-ArgF'/>
                    </Flex>
                </Flex>
                <Text  width="30%" textAlign='justify' paddingLeft={'1em'} >
                    <b>Pablo Argallero Fernández</b>, estudiante de Ingeniería Informática en la Universidad de Oviedo, ha desarrollado este proyecto como Trabajo de Fin de Grado. <br/><br/>
                    El objetivo del proyecto es el estudio de alternativas a la detección de DeepFakes en vídeos, utilizando dos modelos de aprendizaje profundo: Convolutional Neural Networks (CNN) y Recurrent Neural Networks (RNN).<br/><br/>              
                    Para la creación de este proyecto se han realizado labores de investigación, preprocesado de imagenes, entrenamiento de modelos y análisis de resultados. A todo ello, se le ha añadido una interfaz web para facilitar la interacción con los modelos.<br/><br/>
                    El proyecto ha sido dirigido por el profesor <b>Cristian González García</b>. 
                </Text>
            </Flex>
        </Flex>
    )
}

export default About;