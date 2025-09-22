import torch
from pprint import pprint

import torch

class EventTensor:
    def __init__(self, pos_part, neg_part, device=None):
        assert pos_part.shape == neg_part.shape, "Both tensors must have the same shape"
        assert pos_part.dtype == torch.bool and neg_part.dtype == torch.bool, "Both tensors must be boolean"
        
        self.device = device if device else torch.device('cpu')
        self.pos_part = pos_part.to(self.device)
        self.neg_part = neg_part.to(self.device)

    def to_int_tensor(self):
        return self.pos_part.uint8() - self.neg_part.int()

    def to_float_tensor(self):
        return self.pos_part.float() - self.neg_part.float()

    @staticmethod
    def from_values(values, device=None):
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.int, device=device)
        else:
            values = values.to(device)
        pos_part = values == 1
        neg_part = values == -1
        return EventTensor(pos_part, neg_part, device=device)

    @staticmethod
    def zeros(shape, dtype=torch.bool, device=None):
        pos_part = torch.zeros(shape, dtype=dtype, device=device)
        neg_part = torch.zeros(shape, dtype=dtype, device=device)
        return EventTensor(pos_part, neg_part, device=device)

    @staticmethod
    def ones(shape, dtype=torch.bool, device=None):
        pos_part = torch.ones(shape, dtype=dtype, device=device)
        neg_part = torch.zeros(shape, dtype=dtype, device=device)
        return EventTensor(pos_part, neg_part, device=device)

    def to_device(self, device):
        """Transfer EventTensor to a specified device."""
        self.pos_part = self.pos_part.to(device)
        self.neg_part = self.neg_part.to(device)
        self.device = device
        return self

    def index_put_(self, indices, values, accumulate=False):
        if not isinstance(values, EventTensor):
            values = EventTensor.from_values([values], device=self.device).to_int_tensor()
        
        new_pos_part = values > 0
        new_neg_part = values < 0

        if accumulate:
            self.pos_part.index_put_(indices, new_pos_part, accumulate=True)
            self.neg_part.index_put_(indices, new_neg_part, accumulate=True)
        else:
            self.pos_part.index_put_(indices, new_pos_part)
            self.neg_part.index_put_(indices, new_neg_part)

    def sum(self, axis=None):
        return self.to_int_tensor().sum(dim=axis)

    def __getitem__(self, index):
        new_pos_part = self.pos_part[index]
        new_neg_part = self.neg_part[index]
        return EventTensor(new_pos_part, new_neg_part, device=self.device)

    def __setitem__(self, index, value):
        if isinstance(value, EventTensor):
            self.pos_part[index] = value.pos_part.to(self.device)
            self.neg_part[index] = value.neg_part.to(self.device)
        elif isinstance(value, (int, float)):
            value = EventTensor.from_values([value], device=self.device)
            self.pos_part[index] = value.pos_part.squeeze()
            self.neg_part[index] = value.neg_part.squeeze()
        else:
            raise TypeError("Value must be an EventTensor or an integer")

    def __add__(self, other):
        result_tensor = self.to_float_tensor() + other.to_float_tensor()
        return EventTensor.from_values(result_tensor, device=self.device)

    def __sub__(self, other):
        result_tensor = self.to_float_tensor() - other.to_float_tensor()
        return EventTensor.from_values(result_tensor, device=self.device)

    def __mul__(self, other):
        result_tensor = self.to_float_tensor() * other.to_float_tensor()
        return EventTensor.from_values(result_tensor, device=self.device)

    def __truediv__(self, other):
        result_tensor = self.to_float_tensor() / other.to_float_tensor()
        return EventTensor.from_values(result_tensor.round().int(), device=self.device)

    def __repr__(self):
        return f"EventTensor(values=\n{self.to_int_tensor()})"


if __name__ == '__main__':
    # main()

    # Example of how to use the EventTensor class
    data1 = [[1], [-1], [0], [1], [-1], [0]]
    data2 = [0, 1, -1, -1, 0, 1]

    event_tensor1 = EventTensor.from_values(data1)
    event_tensor2 = EventTensor.from_values(data2)

    # Perform operations
    result_add = event_tensor1 + event_tensor2
    result_sub = event_tensor1 - event_tensor2
    result_mul = event_tensor1 * event_tensor2
    result_div = event_tensor1 / event_tensor2

    event_tensor1 = event_tensor1 - event_tensor2


    pprint(event_tensor1)

    # Multiple Indexing examples
    print("Element at index [2, 4]:", event_tensor1[2, 4])  # Example assuming a higher-dimensional tensor
    event_tensor1[0, 0:2] = EventTensor.from_values([0, -1])  # Slice assignment

    # Display the results
    print("Original Event Tensor 1:", event_tensor1)
    print("Original Event Tensor 2:", event_tensor2)
    print("Addition Result:", result_add)
    print("Subtraction Result:", result_sub)
    print("Multiplication Result:", result_mul)
    print("Division Result:", result_div)

