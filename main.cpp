#include <vector>
#include <cstring>
#include <iterator>
#include "mpi.h"

void InitMatrix(int cols, bool* matrix) {
    matrix[2 * cols + 0] = true;
    matrix[2 * cols + 1] = true;
    matrix[2 * cols + 2] = true;
    matrix[1 * cols + 2] = true;
    matrix[0 * cols + 1] = true;
}

void ConfigureOffsetsCounts(int* offsets, int* counts, int rows, int cols, int size) {
    int default_size = rows / size;
    for (int i = 0; i < size; ++i) {
        counts[i] = default_size * cols;
    }
    int left = rows % size;
    for (int i = 0; i < left; ++i) {
        counts[i] += cols;
    }

    offsets[0] = 0;
    for (int i = 1; i < size; ++i) {
        offsets[i] = offsets[i - 1] + counts[i - 1];
    }
}

void PrintMatrix(const bool* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%d", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void ConfigureNeighbours(int size, int rank, int* neighbour_top, int* neighbour_bot) {
    *neighbour_top = (rank + size - 1) % size;
    *neighbour_bot = (rank + 1) % size;
}

bool CompareMatrices(const bool* first, const bool* second, size_t from, size_t to) {
    for (size_t i = from; i < to; ++i) {
        if (first[i] != second[i]) {
            return false;
        }
    }
    return true;
}

void CalculateStopVector(std::vector<bool*> prev_states, bool* stop_vector, bool* matrix, int rows, int cols) {
    size_t vector_size = prev_states.size() - 1;
    auto it = prev_states.begin();
    for (int i = 0; i < vector_size; ++i) {
        stop_vector[i] = CompareMatrices(*it, matrix, cols, cols * (rows + 1));
        it++;
    }
}

bool CheckIsEnd(int rows, int cols, const bool* stop_matrix) {
    for (int i = 0; i < cols; ++i) {
        bool stop = true;
        for (int j = 0; j < rows; ++j) {
            stop &= stop_matrix[j * cols + i];
        }
        if (stop) return true;
    }
    return false;
}

bool MakeDecision(bool prev, int cnt) {
    if (prev) {
        if (cnt < 2  || cnt > 3) return false;
    } else {
        if (cnt == 3) return true;
    }
    return prev;
}

void CalcNext(int rows, int cols, const bool* prev, bool* next) {
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 0; j < cols; ++j) {
            int cnt = 0;
            if (prev[i * cols + (j + 1) % cols]) cnt++;
            if (prev[i * cols + (j + cols - 1) % cols]) cnt++;
            if (prev[(i + 1) * cols + (j + 1) % cols]) cnt++;
            if (prev[(i + 1) * cols + (j + cols - 1) % cols]) cnt++;
            if (prev[(i - 1) * cols + (j + 1) % cols]) cnt++;
            if (prev[(i - 1) * cols + (j + cols - 1) % cols]) cnt++;
            if (prev[(i + 1) * cols + j]) cnt++;
            if (prev[(i - 1) * cols + j]) cnt++;
            next[i * cols + j] = MakeDecision(prev[i * cols + j], cnt);
        }
    }
}

void EmulateGame(int size, int rank, bool* full_matrix, int size_y, int size_x) {
    int* offsets = new int[size];
    int* counts = new int[size];
    ConfigureOffsetsCounts(offsets, counts, size_y, size_x, size);

    bool* matrix = new bool[counts[rank] + size_x * 2];
    bool* my_matrix = matrix + size_x;
    int my_rows = counts[rank] / size_x;

    MPI_Scatterv(full_matrix, counts, offsets, MPI_C_BOOL, my_matrix, counts[rank], MPI_C_BOOL, 0, MPI_COMM_WORLD);

    int neighbour_top, neighbour_bot;
    ConfigureNeighbours(size, rank, &neighbour_top, &neighbour_bot);

    std::vector<bool*> prev_states;

    int iteration = 0;
    bool stop = false;
    while (!stop) {

        // Save iteration matrix
        bool* next_matrix = new bool[counts[rank] + 2 * size_x];
        bool* my_next_matrix = next_matrix + size_x;
        prev_states.push_back(matrix);

        // Start iteration
        iteration++;
        MPI_Request top_send_req, bot_send_req;
        MPI_Request top_rec_req, bot_rec_req;

        MPI_Isend(my_matrix, size_x, MPI_C_BOOL, neighbour_top, 0, MPI_COMM_WORLD, &top_send_req);
        MPI_Isend(my_matrix + counts[rank] - size_x, size_x, MPI_C_BOOL, neighbour_bot, 0, MPI_COMM_WORLD, &bot_send_req);
        MPI_Irecv(matrix, size_x, MPI_C_BOOL, neighbour_top, 0, MPI_COMM_WORLD, &top_rec_req);
        MPI_Irecv(my_matrix + counts[rank], size_x, MPI_C_BOOL, neighbour_bot, 0, MPI_COMM_WORLD, &bot_rec_req);

        // Check
        bool* stop_vector;
        bool* stop_matrix;
        size_t vector_size = prev_states.size() - 1;
        if (vector_size > 1) {
            stop_vector = new bool[vector_size];
            CalculateStopVector(prev_states, stop_vector, matrix, my_rows, size_x);
            stop_matrix = new bool[vector_size* size];
            MPI_Allgather(stop_vector, (int)vector_size, MPI_C_BOOL, stop_matrix, (int)vector_size, MPI_C_BOOL, MPI_COMM_WORLD);
            stop = CheckIsEnd(size, (int)vector_size, stop_matrix);
            delete[] stop_vector;
            delete[] stop_matrix;
        }

        if (stop) break;

        CalcNext(my_rows, size_x, my_matrix, my_next_matrix);
        MPI_Wait(&top_send_req, MPI_STATUSES_IGNORE);
        MPI_Wait(&top_rec_req, MPI_STATUSES_IGNORE);
        CalcNext(3, size_x, matrix, next_matrix);
        MPI_Wait(&bot_send_req, MPI_STATUSES_IGNORE);
        MPI_Wait(&bot_rec_req, MPI_STATUSES_IGNORE);
        CalcNext(3, size_x, my_matrix + (my_rows - 2) * size_x, my_next_matrix + (my_rows - 2) * size_x);

        if (rank == 0) {
            printf("\niteration: %d; matrix\n", iteration);
            PrintMatrix(matrix, my_rows + 2, size_x);

            printf("\niteration: %d; next_matrix\n", iteration);
            PrintMatrix(next_matrix, my_rows + 2, size_x);
        }

        // Switch main matrix -- next iteration
        matrix = next_matrix;
        my_matrix = my_next_matrix;
    }
    if (rank == 0) {
        printf("Result:\niterations: %d\n", iteration - 1);
    }

    int iter = 0;
    for (bool* matrixDump : prev_states) {
        delete[] (matrixDump);
    }

    delete[] offsets;
    delete[] counts;
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank ==0) fprintf(stderr, "Specify exact 2 args:\ncolumns, rows\n");
        MPI_Finalize();
        return 0;
    }

    int cols = std::atoi(argv[1]);
    int rows = std::atoi(argv[2]);

    bool* matrix = nullptr;
    if (rank == 0) {
        matrix = new bool[rows * cols];
        memset(matrix, false, cols * rows);
        InitMatrix(cols, matrix);
    }

    EmulateGame(size, rank, matrix, rows, cols);
    if (rank == 0) delete[] matrix;
    MPI_Finalize();
    return 0;
}
