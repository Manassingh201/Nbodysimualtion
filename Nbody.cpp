#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <string>
#include <filesystem>
#include <ctime>


// Constants
const int NUM_BODIES = 100;
const float G = 0.01f; // Changed from 6.67430e-8f to 0.1f
const float SOFTENING = 0.1f; // To prevent singularities
const float TIME_STEP = 0.01f;
const float BODY_SCALE = 0.05f; // Increased from 0.02f to make particles bigger
const float INITIAL_SPEED_FACTOR = 0.2f; // Increased from 0.05f to make movement more noticeable
const float SPACE_SIZE = 10.0f;
const float SIMULATION_DURATION = 50.0f;
const std::string OUTPUT_FOLDER = "final";
std::ofstream outputFile;      // <-- Output file stream (lowercase 'p')

// Shader sources
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform vec3 color;
    out vec3 fragColor;
    void main() {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        fragColor = color;
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    in vec3 fragColor;
    out vec4 FragColor;
    void main() {
        FragColor = vec4(fragColor, 1.0);
    }
)";

// Body structure
struct Body {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 acceleration;
    float mass;
    glm::vec3 color;
};

// Global variables
std::vector<Body> bodies;
GLuint shaderProgram;
GLuint VAO, VBO;
glm::mat4 projection;
glm::mat4 view;
glm::vec3 cameraPosition = glm::vec3(0.0f, 0.0f, 30.0f);
glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float yaw = -90.0f;
float pitch = 0.0f;
float lastX = 800.0f / 2.0f;
float lastY = 600.0f / 2.0f;
bool firstMouse = true;
bool mousePressed = false;

// Function prototypes
void initialize();
void updateBodies();
void render();
GLuint createSphereVAO();
GLuint compileShaders();
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

// Function to subdivide a triangle
void subdivideTriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, int levels, std::vector<glm::vec3>& outVertices) {
    if (levels == 0) {
        outVertices.push_back(a);
        outVertices.push_back(b);
        outVertices.push_back(c);
        return;
    }
    
    glm::vec3 ab = glm::normalize(a + b);
    glm::vec3 bc = glm::normalize(b + c);
    glm::vec3 ca = glm::normalize(c + a);
    
    subdivideTriangle(a, ab, ca, levels - 1, outVertices);
    subdivideTriangle(b, bc, ab, levels - 1, outVertices);
    subdivideTriangle(c, ca, bc, levels - 1, outVertices);
    subdivideTriangle(ab, bc, ca, levels - 1, outVertices);
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a window
    GLFWwindow* window = glfwCreateWindow(800, 600, "N-Body Simulation", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Compile shaders and create sphere geometry
    shaderProgram = compileShaders();
    VAO = createSphereVAO();

    // Initialize simulation
    initialize();

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
     auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_c); 

    std::stringstream filename_ss;

    if (!std::filesystem::exists(OUTPUT_FOLDER)) {
        std::filesystem::create_directory(OUTPUT_FOLDER);
        std::cout << "Created output directory: " << OUTPUT_FOLDER << std::endl;
    }


   filename_ss << OUTPUT_FOLDER << "/" // Add subfolder path
                << "nbody_simulation_data_"
                << std::put_time(&now_tm, "%Y%m%d_%H%M%S")
                << ".csv";
    std::string outputFilename = filename_ss.str(); 
    // Use lowercase 'p'
    std::cout << "IMPORTANT: Ensure the output directory '" << OUTPUT_FOLDER << "' exists relative to the executable." << std::endl;
    std::cout << "Attempting to open output file: " << outputFilename << std::endl;

    outputFile.open(outputFilename);
    
    if(!outputFile.is_open()){
        std::cerr << "ERROR: COULD not open the output file" << outputFilename <<std::endl;
        glDeleteVertexArrays(1, &VAO);
        glDeleteProgram(shaderProgram);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    outputFile << std::fixed << std::setprecision(4);
    outputFile << "Time, BodyID, PosX, PosY, PosZ, VelX, VelY, VelZ, Mass\n";
    std::cout << "starting simualtion; recording data to " << outputFilename << "for "<<SIMULATION_DURATION << " simulation time units. " << std::endl;
    // Main loop
    float simulationTime = 0.0f;
    while (!glfwWindowShouldClose(window) && simulationTime < SIMULATION_DURATION) {
        // Process input
        processInput(window);

        // Update physics
        updateBodies();
        for(int i = 0; i<NUM_BODIES; ++i){
            outputFile << simulationTime << ","
            <<i<< ","
            <<bodies[i].position.x << "," <<bodies[i].position.y<<","<<bodies[i].position.z<<","
            <<bodies[i].velocity.x << "," <<bodies[i].velocity.y<<","<<bodies[i].position.z<<","
            <<bodies[i].mass<< "\n";
        }

        simulationTime+=TIME_STEP;

       // std::cout<<simulationTime<<std::endl;
        // Render
        glClearColor(0.0f, 0.0f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        render();

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    if(outputFile.is_open()){
        outputFile.close();
        std::cout<< "simualtion finished final simulation time " << simulationTime << "Data saved" <<std::endl;
    }else{
        std::cerr << "out put file havent open during simualtion"<<std::endl;
    }
    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
    return 0;
}

void initialize() {
    // Set up projection and view matrices
    projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
    view = glm::lookAt(
        glm::vec3(0.0f, 0.0f, 30.0f),  // Camera position
        glm::vec3(0.0f, 0.0f, 0.0f),   // Look at origin
        glm::vec3(0.0f, 1.0f, 0.0f)    // Up vector
    );

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> posDist(-SPACE_SIZE, SPACE_SIZE);
    std::uniform_real_distribution<float> massDist(1.0f, 10.0f);
    std::uniform_real_distribution<float> colorDist(0.3f, 1.0f);

    // Initialize bodies
    bodies.resize(NUM_BODIES);
    
    // Create a central massive body (like a star)
    // bodies[0].position = glm::vec3(0.0f, 0.0f, 0.0f);
    // bodies[0].velocity = glm::vec3(0.0f);
    // bodies[0].acceleration = glm::vec3(0.0f);
    // bodies[0].mass = 10.0f;  // Much more massive
    // bodies[0].color = glm::vec3(1.0f, 0.8f, 0.0f);  // Yellow for the star
    
    // Create other bodies
    for (int i = 0; i < NUM_BODIES; i++) {
        // Position - create a mix of close and distant bodies
        float angle = 2.0f * 3.14159f * static_cast<float>(i) / (NUM_BODIES - 1);
        
        // Create a mix of close and distant bodies
        float distance;
        if (i % 3 == 0) {
            // Close bodies
            distance = 2.0f + posDist(gen) * 0.5f;
        } else if (i % 3 == 1) {
            // Medium distance bodies
            distance = 5.0f + posDist(gen) * 1.0f;
        } else {
            // Distant bodies
            distance = 8.0f + posDist(gen) * 2.0f;
        }
        
        bodies[i].position = glm::vec3(
            distance * cos(angle),
            posDist(gen) * 0.5f,  // Increased vertical deviation
            distance * sin(angle)
        );
        
        // Initial velocity - perpendicular to position for orbit
        // glm::vec3 orbitDir = glm::normalize(glm::cross(bodies[i].position, glm::vec3(0.0f, 1.0f, 0.0f)));
        // float orbitSpeed = INITIAL_SPEED_FACTOR * sqrt((G * bodies[0].mass) / glm::length(bodies[i].position));
        // // bodies[i].velocity = orbitDir * orbitSpeed;
        bodies[i].velocity = glm::vec3(0.0f, 0.0f, 0.0f);
        
        // Initialize other properties
        bodies[i].acceleration = glm::vec3(0.0f);
        bodies[i].mass = 10.0f;  // All particles have the same mass
        bodies[i].color = glm::vec3(colorDist(gen), colorDist(gen), colorDist(gen));
    }
}

void updateBodies() {
    // Calculate accelerations based on gravitational forces
    for (int i = 0; i < NUM_BODIES; i++) {
        bodies[i].acceleration = glm::vec3(0.0f);
        
        for (int j = 0; j < NUM_BODIES; j++) {
            if (i == j) continue;
            
            glm::vec3 r = bodies[j].position - bodies[i].position;
            float distance = glm::length(r) + SOFTENING;
            float forceMagnitude = G * bodies[j].mass / (distance * distance);
            
            bodies[i].acceleration += forceMagnitude * glm::normalize(r);
        }
    }
    
    // Update positions and velocities using Verlet integration
    for (int i = 0; i < NUM_BODIES; i++) {
        bodies[i].velocity += bodies[i].acceleration * TIME_STEP;
        bodies[i].position += bodies[i].velocity * TIME_STEP;
    }
}

void render() {
    glUseProgram(shaderProgram);
    
    // Update view matrix based on camera position and target
    view = glm::lookAt(cameraPosition, cameraPosition + cameraTarget, cameraUp);
    
    // Set view and projection matrices
    GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
    GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    
    // Draw each body
    GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
    GLint colorLoc = glGetUniformLocation(shaderProgram, "color");
    
    for (int i = 0; i < NUM_BODIES; i++) {
        // Calculate scale based on mass
        float scale = BODY_SCALE * std::cbrt(bodies[i].mass);
        
        // Create model matrix
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, bodies[i].position);
        model = glm::scale(model, glm::vec3(scale));
        
        // Set uniforms and draw
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniform3fv(colorLoc, 1, glm::value_ptr(bodies[i].color));
        
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 1080); // Assuming the sphere has 1080 vertices
        glBindVertexArray(0);
    }
}

GLuint createSphereVAO() {
    // Create a simple sphere with icosahedron subdivision
    std::vector<glm::vec3> vertices;
    
    // Start with an icosahedron
    const float X = 0.525731f;
    const float Z = 0.850651f;
    
    vertices.push_back(glm::vec3(-X, 0.0f, Z));
    vertices.push_back(glm::vec3(X, 0.0f, Z));
    vertices.push_back(glm::vec3(-X, 0.0f, -Z));
    vertices.push_back(glm::vec3(X, 0.0f, -Z));
    vertices.push_back(glm::vec3(0.0f, Z, X));
    vertices.push_back(glm::vec3(0.0f, Z, -X));
    vertices.push_back(glm::vec3(0.0f, -Z, X));
    vertices.push_back(glm::vec3(0.0f, -Z, -X));
    vertices.push_back(glm::vec3(Z, X, 0.0f));
    vertices.push_back(glm::vec3(-Z, X, 0.0f));
    vertices.push_back(glm::vec3(Z, -X, 0.0f));
    vertices.push_back(glm::vec3(-Z, -X, 0.0f));
    
    // Define indices for the faces of the icosahedron
    std::vector<glm::ivec3> indices = {
        {0, 4, 1}, {0, 9, 4}, {9, 5, 4}, {4, 5, 8}, {4, 8, 1},
        {8, 10, 1}, {8, 3, 10}, {5, 3, 8}, {5, 2, 3}, {2, 7, 3},
        {7, 10, 3}, {7, 6, 10}, {7, 11, 6}, {11, 0, 6}, {0, 1, 6},
        {6, 1, 10}, {9, 0, 11}, {9, 11, 2}, {9, 2, 5}, {7, 2, 11}
    };
    
    // Subdivide the icosahedron to make a sphere
    std::vector<glm::vec3> sphereVertices;
    for (const auto& face : indices) {
        subdivideTriangle(vertices[face.x], vertices[face.y], vertices[face.z], 2, sphereVertices);
    }
    
    // Create the VAO and VBO
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    
    glBindVertexArray(vao);
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sphereVertices.size() * sizeof(glm::vec3), sphereVertices.data(), GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    return vao;
}

GLuint compileShaders() {
    // Compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    // Check for errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "Vertex shader compilation failed:\n" << infoLog << std::endl;
    }
    
    // Compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    
    // Check for errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Fragment shader compilation failed:\n" << infoLog << std::endl;
    }
    
    // Link shaders
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    // Check for errors
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "Shader program linking failed:\n" << infoLog << std::endl;
    }
    
    // Clean up
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return program;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
        
    // Camera controls
    float cameraSpeed = 0.5f;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPosition += cameraSpeed * cameraTarget;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPosition -= cameraSpeed * cameraTarget;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPosition -= glm::normalize(glm::cross(cameraTarget, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPosition += glm::normalize(glm::cross(cameraTarget, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        cameraPosition -= cameraUp * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        cameraPosition += cameraUp * cameraSpeed;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraTarget = glm::normalize(front);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        mousePressed = true;
    } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        mousePressed = false;
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    // Zoom in/out by moving the camera closer to or further from the target
    float zoomSpeed = 1.0f;
    cameraPosition += cameraTarget * float(yoffset) * zoomSpeed;
}