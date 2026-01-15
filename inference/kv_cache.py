class SequenceKVCache:
    def __init__(self, allocator):
        self.allocator = allocator
        self.pages = []
        self.cur_slot = 0

    def append(self, k_i8, v_fp16, scale):
        if self.cur_slot == 0:
            page = self.allocator.alloc_page()
            self.pages.append(page)

        page_id = self.pages[-1]
        self.allocator.write_kv(
            page_id,
            self.cur_slot,
            k_i8,
            v_fp16,
            scale
        )

        self.cur_slot += 1
        if self.cur_slot == self.allocator.page_size:
            self.cur_slot = 0

    def get_kv(self):
        return self.allocator.read_kv(self.pages)
