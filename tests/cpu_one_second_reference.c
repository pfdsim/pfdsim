/*
 * Single-core CPU throughput reference for PFDsim performance tests.
 * Compile with: cc -O3 -march=native -o cpu_reference this_file.c
 *
 * This fixed workload takes approximately one second on the 2026-08-08
 * reference CPU. It deliberately does not calibrate itself at runtime.
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define REFERENCE_ITERATIONS UINT64_C(324000000)

static volatile uint64_t saved_integer;
static volatile double saved_floating_point;

static double monotonic_seconds(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return -1.0;
    }
    return (double)now.tv_sec + (double)now.tv_nsec * 1.0e-9;
}

int main(void)
{
    uint64_t state = UINT64_C(0x9e3779b97f4a7c15);
    uint64_t mixed = UINT64_C(0xd1b54a32d192ed03);
    double x = 0.625;
    double y = 0.375;
    const double started = monotonic_seconds();

    for (uint64_t index = 0; index < REFERENCE_ITERATIONS; ++index) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        mixed += state ^ (mixed << 9) ^ (mixed >> 11);

        x = x * 1.00000011920928955078125
            + (double)(state & UINT64_C(1023)) * 0x1p-30;
        if (x > 4.0) {
            x -= 3.0;
        }
        y = y * 0.999999940395355224609375 + x * 0x1p-20;
        if (y > 2.0) {
            y -= 1.0;
        }
    }

    const double finished = monotonic_seconds();
    saved_integer = state ^ mixed;
    saved_floating_point = x + y;

    if (started < 0.0 || finished < 0.0) {
        fputs("clock_gettime failed\n", stderr);
        return 2;
    }

    printf("elapsed_seconds=%.9f iterations=%" PRIu64
           " checksum=%016" PRIx64 "/%.17g\n",
           finished - started,
           REFERENCE_ITERATIONS,
           saved_integer,
           saved_floating_point);
    return 0;
}
