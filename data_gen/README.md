## Uni-Edit Data Construction

**1. Data Initialization**
Reformulate the format of your VQA data.
```bash
python first_process.py
```

> **1.1 Example**
> One example of `llava_onevision_1.5_instruct_22M.jsonl`.
> ```bash
> python first_process_ov_example.py
> ```

**2. QA Classification & Processing**
Using gpt-4o to classify the QA type and process the answer.
```bash
python second_process.py
```

> **2.1 Filter**
> Balancing the QA types.
> ```bash
> python second_process_filter.py
> ```

**3. Instruction Generation**
Changing the question into the intelligent edit instruction.
```bash
python third_process.py
```

**4. Image Generation**
Generating the edited images using nano-pro.
```bash
python fourth_process.py
```

**5. Data Finalization**
Building the jsonl in `edit` folder and remove the ones with duplicated image names.
```bash
python fifth_process.py
```

> **5.1 Filter**
> Removing the poor visual effect samples.
> ```bash
> python fifth_process_filter.py
> ```