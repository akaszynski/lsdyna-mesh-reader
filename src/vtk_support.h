#ifndef VTK_SUPPORT_H
#define VTK_SUPPORT_H

#include <stdint.h>

int ans_to_vtk(
    const int,
    const int *,
    const int *,
    const int *,
    const int,
    const int *,
    int64_t *,
    int64_t *,
    uint8_t *);

#endif
