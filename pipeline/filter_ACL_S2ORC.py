import json

# feel free to wrap this into a larger loop for batches 0~99
#BATCH_ID = 0
for BATCH_ID in range(1,100):
    print(BATCH_ID)
    # filter papers using metadata values
    filtered_paper_id = []
    with open(f'20200705v1/full/metadata/metadata_{BATCH_ID}.jsonl') as f_meta:
        for line in f_meta:
            metadata_dict = json.loads(line)
            paper_id = metadata_dict['paper_id']
            #print(f"Currently viewing S2ORC paper: {paper_id}")

            # suppose we only care about ACL anthology papers
            if not metadata_dict['acl_id']:
                continue

            # and we want only papers with resolved outbound citations
            if not metadata_dict['has_outbound_citations']:
                continue

            filtered_paper_id.append(paper_id)

    filtered_paper_id = set(filtered_paper_id)

    with open("20200705v1/acl/metadata.jsonl", "a") as acl_meta:
        with open(f'20200705v1/full/metadata/metadata_{BATCH_ID}.jsonl') as f_meta:
            for line in f_meta:
                metadata_dict = json.loads(line)
                if metadata_dict['paper_id'] in filtered_paper_id:
                    acl_meta.write(line)

    # create a lookup for the pdf parse based on paper ID
    #paper_id_to_pdf_parse = {}
    with open("20200705v1/acl/pdf_parses.jsonl", "a") as acl_pdf:
        with open(f'20200705v1/full/pdf_parses/pdf_parses_{BATCH_ID}.jsonl') as f_pdf:
            for line in f_pdf:
                pdf_parse_dict = json.loads(line)
                if pdf_parse_dict['paper_id'] in filtered_paper_id:
                    acl_pdf.write(line)
                #paper_id_to_pdf_parse[pdf_parse_dict['paper_id']] = pdf_parse_dict
