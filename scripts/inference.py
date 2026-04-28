import os
import yaml
import torch
import re 
from unsloth import FastLanguageModel

class IntentClassification:
    def __init__(self, config_path):
        """
        Khởi tạo class suy luận.
        :param config_path: Đường dẫn tới file configs/inference.yaml 
        """
        # 1. Đọc cấu hình từ file YAML
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)['inference']
        
        self.model_path = config['model_path']
        self.max_seq_length = config.get('max_seq_length', 512)
        self.load_in_4bit = config.get('load_in_4bit', True)
        self.device = config.get('device', 'cuda')

        print(f"Đang tải mô hình từ checkpoint: {self.model_path}...")
        
        # 2. Tải mô hình và tokenizer (sử dụng Unsloth để tăng tốc suy luận)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_path,
            max_seq_length = self.max_seq_length,
            dtype = None,
            load_in_4bit = self.load_in_4bit,
        )
        
        # Kích hoạt chế độ suy luận nhanh của Unsloth
        FastLanguageModel.for_inference(self.model)

        labels_str = "activate_my_card, age_limit, apple_pay_or_google_pay, atm_support, automatic_top_up, balance_not_updated_after_bank_transfer, balance_not_updated_after_cheque_or_cash_deposit, beneficiary_not_allowed, cancel_transfer, card_about_to_expire, card_acceptance, card_arrival, card_delivery_estimate, card_linking, card_not_working, card_payment_fee_charged, card_payment_not_recognised, card_payment_wrong_exchange_rate, card_swallowed, cash_withdrawal_charge, cash_withdrawal_not_recognised, change_pin, compromised_card, contactless_not_working, country_support, declined_card_payment, declined_cash_withdrawal, declined_transfer, direct_debit_payment_not_recognised, disposable_card_limits, edit_personal_details, exchange_charge, exchange_rate, exchange_via_app, extra_charge_on_statement, failed_transfer, fiat_currency_support, get_disposable_virtual_card, get_physical_card, getting_spare_card, getting_virtual_card, lost_or_stolen_card, lost_or_stolen_phone, order_physical_card, passcode_forgotten, pending_card_payment, pending_cash_withdrawal, pending_top_up, pending_transfer, pin_blocked, receiving_money, Refund_not_showing_up, request_refund, reverted_card_payment?, supported_cards_and_currencies, terminate_account, top_up_by_bank_transfer_charge, top_up_by_card_charge, top_up_by_cash_or_cheque, top_up_failed, top_up_limits, top_up_reverted, topping_up_by_card, transaction_charged_twice, transfer_fee_charged, transfer_into_account, transfer_not_received_by_recipient, transfer_timing, unable_to_verify_identity, verify_my_identity, verify_source_of_funds, verify_top_up, virtual_card_not_working, visa_or_mastercard, why_verify_identity, wrong_amount_of_cash_received, wrong_exchange_rate_for_cash_withdrawal"
        
        # Template prompt phải trùng khớp với lúc huấn luyện
        self.prompt_template = (
            "### Instruction:\n"
            f"Classify the banking message into ONE of these categories: [{labels_str}]. Output ONLY the label name. Do not invent new labels.\n\n"
            "### Input:\n"
            "{message}\n\n"
            "### Response:\n"
        )

        self.valid_labels = ["activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support", "automatic_top_up", "balance_not_updated_after_bank_transfer", "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed", "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival", "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged", "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin", "compromised_card", "contactless_not_working", "country_support", "declined_card_payment", "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised", "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app", "extra_charge_on_statement", "failed_transfer", "fiat_currency_support", "get_disposable_virtual_card", "get_physical_card", "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked", "receiving_money", "Refund_not_showing_up", "request_refund", "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account", "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted", "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged", "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing", "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds", "verify_top_up", "virtual_card_not_working", "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal"]
        
    def __call__(self, message):
        """
        Nhận đầu vào là tin nhắn và trả về nhãn dự đoán
        """

        # Định dạng input theo template
        prompt = self.prompt_template.format(message=message)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # Thực hiện suy luận
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=32,
                use_cache=True
            )

        # Giải mã kết quả (chỉ lấy phần nội dung sau "### Response:")
        decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        predicted = decoded_output.split("### Response:")[-1].strip().lower() 

        if prediction not in self.valid_labels:
            closest_match = difflib.get_close_matches(raw_prediction, self.valid_labels, n=1, cutoff=0.0)
            return closest_match[0] if closest_match else prediction
        
        return prediction 

if __name__ == "__main__":
    # Đường dẫn tới file cấu hình suy luận
    config_file = os.path.join(os.path.dirname(__file__), "..", "configs", "inference.yaml")
    
    # Khởi tạo module
    classifier = IntentClassification(config_file)

    # Thử nghiệm với 3 tin nhắn mẫu trong tập test 
    test_messages = ["How do I link this new card?", "How do I retrieve my card from the machine?", "I want to know where the funds come from."] 
    for test_message in test_messages:
        prediction = classifier(test_message)
        print(f"Message: {test_message}")
        print(f"Predicted Intent: {prediction}")
