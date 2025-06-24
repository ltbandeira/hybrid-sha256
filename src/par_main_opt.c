#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "sha256.h"

#define PREFIX "desafio"
#define TAG_PEDIDO 1
#define TAG_TRABALHO 2
#define TAG_SOLUCAO 3
#define TAG_PARADA 4

#define STEP_SIZE 40000ULL
#define MAX_MSG_LEN 256

void hash_to_hex(uint8_t hash[SHA256_BLOCK_SIZE], char hex_output[SHA256_BLOCK_SIZE * 2 + 1])
{
    for (int i = 0; i < SHA256_BLOCK_SIZE; i++)
        sprintf(hex_output + (i * 2), "%02x", hash[i]);
    hex_output[SHA256_BLOCK_SIZE * 2] = '\0';
}

int hash_has_n_zero_bits(uint8_t hash[SHA256_BLOCK_SIZE], int n)
{
    int full_bytes = n / 8;
    int remaining_bits = n % 8;
    for (int i = 0; i < full_bytes; ++i)
        if (hash[i] != 0)
            return 0;
    if (remaining_bits > 0)
    {
        uint8_t mask = 0xFF << (8 - remaining_bits);
        if ((hash[full_bytes] & mask) != 0)
            return 0;
    }
    return 1;
}

void compute_sha256(const char *input, uint8_t output[SHA256_BLOCK_SIZE])
{
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (const uint8_t *)input, strlen(input));
    sha256_final(&ctx, output);
}

int main(int argc, char **argv)
{
    int rank, size, provided;
    double t_start, t_end;

    MPI_Status status;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_MULTIPLE)
    {
        if (rank == 0)
            fprintf(stderr,
                    "Erro: MPI não fornece MPI_THREAD_MULTIPLE (forneceu %d)\n",
                    provided);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* -------- leitura dos argumentos -------- */
    if (argc < 4)
    {
        if (rank == 0)
            fprintf(stderr,
                    "Uso: mpirun -np <procs> %s <numero_solucoes> <bits_zero> "
                    "<threads_openmp>\n",
                    argv[0]);
        MPI_Finalize();
        return 1;
    }
    int total_solucoes = atoi(argv[1]);
    int target_zero_bits = atoi(argv[2]);
    int num_threads_cli = atoi(argv[3]);

    /* -------- descoberta do hostname do mestre -------- */
    char my_host[MPI_MAX_PROCESSOR_NAME];
    char master_host[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(my_host, &name_len);

    if (rank == 0)
        strcpy(master_host, my_host);

    MPI_Bcast(master_host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
              0, MPI_COMM_WORLD);

    /* -------- decide nº de threads OpenMP -------- */
    int threads_ativas;

    if (rank == 0)
    {
        threads_ativas = 1;
    }
    else
    {
        threads_ativas =
            (strcmp(my_host, master_host) == 0)
                ? num_threads_cli - 1
                : num_threads_cli;
        if (threads_ativas < 1)
            threads_ativas = 1;
    }
    omp_set_num_threads(threads_ativas);

    t_start = MPI_Wtime();

    if (rank == 0)
    {
        // Mestre
        unsigned long long next_start = 0;
        int stop_signal = 1, found = 0;

        // Alocação de arrays para índice e string
        unsigned long long *indices = malloc(total_solucoes * sizeof *indices);
        char **sols = malloc(total_solucoes * sizeof *sols);
        for (int i = 0; i < total_solucoes; i++)
            sols[i] = malloc(MAX_MSG_LEN);

        while (found < total_solucoes)
        {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_PEDIDO)
            {
                unsigned long long step;
                MPI_Recv(&step, 1, MPI_UNSIGNED_LONG_LONG,
                         status.MPI_SOURCE, TAG_PEDIDO, MPI_COMM_WORLD, &status);
                MPI_Send(&next_start, 1, MPI_UNSIGNED_LONG_LONG,
                         status.MPI_SOURCE, TAG_TRABALHO, MPI_COMM_WORLD);
                next_start += step;
            }
            else if (status.MPI_TAG == TAG_SOLUCAO)
            {
                char msg[2][MAX_MSG_LEN];
                MPI_Recv(msg, 2 * MAX_MSG_LEN, MPI_CHAR,
                         status.MPI_SOURCE, TAG_SOLUCAO, MPI_COMM_WORLD, &status);

                indices[found] = strtoull(msg[1], NULL, 10);
                strncpy(sols[found], msg[0], MAX_MSG_LEN - 1);
                sols[found][MAX_MSG_LEN - 1] = '\0';

                printf("[MESTRE] %2d) \"%s\" (idx=%llu) do worker %d\n",
                       found + 1, sols[found], indices[found], status.MPI_SOURCE);
                found++;
            }
        }

        // Sinaliza parada
        for (int w = 1; w < size; w++)
            MPI_Send(&stop_signal, 1, MPI_INT, w, TAG_PARADA, MPI_COMM_WORLD);

        // Ordena resultados pelo índice
        for (int i = 0; i < total_solucoes - 1; i++)
            for (int j = i + 1; j < total_solucoes; j++)
                if (indices[i] > indices[j])
                {
                    unsigned long long tmp_idx = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp_idx;
                    char *tmp = sols[i];
                    sols[i] = sols[j];
                    sols[j] = tmp;
                }

        printf("\n[MESTRE] Soluções ordenadas:\n");
        for (int i = 0; i < total_solucoes; i++)
            printf("  %2d) %s  (idx=%llu)\n", i + 1, sols[i], indices[i]);

        for (int i = 0; i < total_solucoes; i++)
            free(sols[i]);
        free(sols);
        free(indices);
    }
    else
    {
        // Trabalhadores
        int stop = 0;

        while (!stop)
        {
            // Verifica parada antes de pedir trabalho
            int flag;
            MPI_Iprobe(0, TAG_PARADA, MPI_COMM_WORLD, &flag, &status);
            if (flag)
            {
                int dummy;
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_PARADA,
                         MPI_COMM_WORLD, &status);
                break;
            }

            // Pede trabalho ao mestre
            unsigned long long step = STEP_SIZE;
            MPI_Send(&step, 1, MPI_UNSIGNED_LONG_LONG, 0, TAG_PEDIDO, MPI_COMM_WORLD);

            // Espera por trabalho ou por parada
            while (1)
            {
                MPI_Iprobe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
                if (!flag)
                    continue;

                if (status.MPI_TAG == TAG_PARADA)
                {
                    int dummy;
                    MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_PARADA,
                             MPI_COMM_WORLD, &status);
                    stop = 1;
                    break;
                }
                else if (status.MPI_TAG == TAG_TRABALHO)
                {
                    unsigned long long start;
                    MPI_Recv(&start, 1, MPI_UNSIGNED_LONG_LONG, 0, TAG_TRABALHO, MPI_COMM_WORLD, &status);

// Processa o trabalho recebido
#pragma omp parallel for schedule(dynamic)
                    for (unsigned long long i = start; i < start + step; i++)
                    {
                        uint8_t hash[SHA256_BLOCK_SIZE];
                        char txt[1024];
                        snprintf(txt, sizeof(txt), PREFIX "%llu", i);
                        compute_sha256(txt, hash);

                        // Prepara mensagem com índice e string
                        if (hash_has_n_zero_bits(hash, target_zero_bits))
                        {
                            char msg[2][MAX_MSG_LEN];
                            snprintf(msg[0], MAX_MSG_LEN, "%s", txt);
                            snprintf(msg[1], MAX_MSG_LEN, "%llu", i);
                            int tid = omp_get_thread_num();
                            printf("[Worker %d / thread %d] achou: %s \n", rank, tid, txt);
                            MPI_Send(msg, 2 * MAX_MSG_LEN, MPI_CHAR, 0, TAG_SOLUCAO, MPI_COMM_WORLD);
                        }
                    }
                    break;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    t_end = MPI_Wtime() - t_start;

    if (rank == 0)
    {
        printf("%d,%f\n", size, t_end); // saída salva em formato CSV (número de processos, tempo total)
    }

    MPI_Finalize();
    return 0;
}
