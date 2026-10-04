"""
scripts/generate_synthetic_docs_v2.py
======================================
Generates 5 additional documents for the V2 dataset.
V2 = V1 (22 docs) + 5 new docs = 27 total.

Demonstrates DVC dataset versioning: V1 -> V2.

Run AFTER generate_synthetic_docs.py (which creates V1):
    python scripts/generate_synthetic_docs_v2.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

V2_DOCUMENTS = [
    (
        "HR-PROMO-001",
        "hr/promotion_policy.txt",
        "HR",
        "Career Development",
        "1.0",
        "2026-01-01",
        "internal_policy",
        "employee",
        "Promotion and Career Progression Policy",
        """PROMOTION AND CAREER PROGRESSION POLICY
Version 1.0 | Effective Date: 1 January 2026 | Document ID: HR-PROMO-001
Department: Human Resources | Category: Career Development

1. PURPOSE
This policy defines the criteria, process, and timeline for employee
promotions and career progression at Acme Corp.

2. ELIGIBILITY
2.1 Employees must have completed at least 18 months in their current role.
2.2 The last performance rating must be "Meets Expectations" or higher.
2.3 Employees under a Performance Improvement Plan (PIP) are not eligible
    for promotion until the PIP is successfully closed.

3. PROMOTION CYCLE
3.1 Promotions are reviewed twice a year: in April (mid-year) and October (year-end).
3.2 Off-cycle promotions may be approved by the HR Director and Business Head
    for exceptional cases such as critical retention risk or significant role expansion.

4. PROMOTION CRITERIA
4.1 Sustained performance above role expectations for at least 2 consecutive cycles.
4.2 Demonstrated readiness for the next level based on the competency framework.
4.3 Business need for the higher role (no promotion without a justified business case).
4.4 Manager recommendation with documented justification.

5. PROCESS
5.1 Manager nominates the employee via the HR portal by the nomination deadline.
5.2 HR reviews nominations against eligibility criteria.
5.3 A calibration committee (manager + HRBP + Business Head) approves promotions.
5.4 Employees are notified of outcomes within 5 working days of the calibration meeting.

6. COMPENSATION IMPACT
6.1 Promotions are accompanied by a salary increase of 15% to 25% depending
    on the band jump and market benchmarking.
6.2 The new compensation is effective from the 1st of the month following promotion approval.

7. POLICY OWNER
Human Resources — Talent Management. Contact hr@acmecorp.internal.
""",
    ),

    (
        "IT-ACCESS-001",
        "it/access_control_policy.txt",
        "IT",
        "Security",
        "2.0",
        "2026-01-01",
        "internal_policy",
        "employee",
        "Access Control Policy",
        """ACCESS CONTROL POLICY
Version 2.0 | Effective Date: 1 January 2026 | Document ID: IT-ACCESS-001
Department: IT | Category: Security

1. PURPOSE
To ensure that access to Acme Corp systems and data is granted on the
principle of least privilege and is regularly reviewed.

2. PRINCIPLE OF LEAST PRIVILEGE
2.1 Employees are granted access only to the systems and data required to
    perform their job responsibilities.
2.2 Default access is no access — access must be explicitly provisioned.

3. ACCESS PROVISIONING
3.1 New employee access is provisioned based on the IT Access Request submitted
    by HR during onboarding.
3.2 Role-based access templates are pre-defined for standard job functions.
3.3 Elevated or privileged access requires additional approval from the
    IT Security team and the employee's department head.

4. ACCESS REVIEW
4.1 Access rights are reviewed quarterly by IT Security in coordination with managers.
4.2 Managers must certify or revoke their team members' access during each review.
4.3 Dormant accounts (no login for 60 days) are automatically disabled.

5. ACCESS REMOVAL
5.1 Upon resignation or termination, all access is revoked within 24 hours
    of the HR notification to IT.
5.2 For involuntary terminations, access is revoked immediately upon HR notification.

6. PRIVILEGED ACCESS MANAGEMENT
6.1 Admin and service account credentials are managed via the PAM system (CyberArk).
6.2 Privileged sessions are recorded and may be audited.

7. THIRD-PARTY ACCESS
7.1 Vendors and contractors are granted time-bound access reviewed every 90 days.
7.2 Third-party access must be approved by the IT Security team and the sponsoring
    business manager.

8. POLICY OWNER
IT Security. Contact security@acmecorp.internal.
""",
    ),

    (
        "TRAVEL-INTL-001",
        "travel/international_travel_policy.txt",
        "Travel",
        "International Travel",
        "1.1",
        "2026-01-01",
        "internal_policy",
        "employee",
        "International Travel Policy",
        """INTERNATIONAL TRAVEL POLICY
Version 1.1 | Effective Date: 1 January 2026 | Document ID: TRAVEL-INTL-001
Department: Travel Management | Category: International Travel

1. PURPOSE
This policy supplements TRAVEL-POL-001 with specific guidelines for
international business travel on behalf of Acme Corp.

2. APPROVAL REQUIREMENTS
2.1 International travel requires approval from the employee's direct manager
    and the HR Director at least 15 working days before departure.
2.2 Travel to countries with Government of India "Do Not Travel" advisories
    requires additional approval from the Chief Risk Officer.

3. VISA AND DOCUMENTATION
3.1 Employees are responsible for obtaining the correct visa with at least
    3 working days before the visa application submission date.
3.2 The Travel Desk assists with invitation letters and support documentation.
3.3 Passport validity must be at least 6 months beyond the return date.

4. HEALTH AND VACCINATIONS
4.1 Employees must check vaccination requirements for the destination country.
4.2 Required vaccinations are reimbursable per HR-REIMB-001.
4.3 Travel health advisories must be checked at mohfw.gov.in before departure.

5. TRAVEL CLASS AND ACCOMMODATION
5.1 Flights under 6 hours: economy class.
5.2 Flights over 8 hours: business class permitted with Director approval.
5.3 Hotel rate limit: USD 150/night (or local equivalent).
5.4 Long-stay accommodation (over 14 nights): serviced apartments may be
    arranged by the Travel Desk if cost-effective.

6. PER DIEM (INTERNATIONAL)
6.1 Per diem rates are destination-specific and published in the International
    Per Diem Table at intranet.acmecorp.internal/travel/per-diem-international.
6.2 Per diem covers meals and incidentals. Accommodation is booked separately.

7. SAFETY AND SECURITY
7.1 Employees must register international trips with the Travel Desk before departure.
7.2 The Travel Desk monitors traveller safety via the travel risk management system.
7.3 In case of security emergency abroad: contact +91-1800-XXX-XXXX (24x7).

8. FOREX AND EXPENSES
8.1 Use the company forex card loaded by Finance for international expenses.
8.2 Cash withdrawals above USD 200/day require manager pre-approval.

9. POLICY OWNER
Finance — Travel Desk. Contact travel-desk@acmecorp.internal.
""",
    ),

    (
        "HR-PROBA-001",
        "hr/probation_policy.txt",
        "HR",
        "Employment",
        "2.0",
        "2026-01-01",
        "internal_policy",
        "employee",
        "Probation Policy",
        """PROBATION POLICY
Version 2.0 | Effective Date: 1 January 2026 | Document ID: HR-PROBA-001
Department: Human Resources | Category: Employment

1. PURPOSE
This policy defines the probationary period for new employees at Acme Corp
and the criteria for successful confirmation.

2. PROBATIONARY PERIOD
2.1 All new permanent employees serve a probationary period of 6 months
    from their date of joining.
2.2 In exceptional cases, the probationary period may be extended for up to
    3 additional months with written approval from the HR Director.

3. BENEFITS DURING PROBATION
3.1 Probationary employees are entitled to all standard benefits from day one
    (health insurance, provident fund, etc.) as per HR-BENEFITS-001.
3.2 Annual leave accrues during the probationary period but may not be
    consumed in the first 3 months (refer to HR-LEAVE-001).
3.3 WFH is not available during the probationary period (refer to HR-WFH-001).

4. PERFORMANCE REVIEW
4.1 A formal probation review is conducted at 3 months and at 6 months.
4.2 The manager completes a structured assessment form covering: quality of
    work, collaboration, communication, and adherence to company values.
4.3 The employee receives written feedback after each review.

5. CONFIRMATION
5.1 Upon successful completion of the 6-month probationary period, the employee
    receives a written confirmation letter.
5.2 The confirmation is effective from the first day after the probationary period.

6. TERMINATION DURING PROBATION
6.1 Either party may terminate employment during the probationary period
    with 1 month's notice or payment in lieu of notice.
6.2 The company may terminate employment with immediate effect in cases
    of gross misconduct.

7. POLICY OWNER
Human Resources Department. Contact hr@acmecorp.internal.
""",
    ),

    (
        "PROD-WARRANTY-001",
        "product/warranty_policy.txt",
        "Product",
        "Product Information",
        "1.2",
        "2026-01-01",
        "internal_policy",
        "employee",
        "AcmeSuite Warranty and SLA Policy",
        """ACMESUITE WARRANTY AND SLA POLICY
Version 1.2 | Effective Date: 1 January 2026 | Document ID: PROD-WARRANTY-001
Department: Product | Category: Product Information

1. PURPOSE
This document defines the warranty commitments and Service Level Agreements
(SLAs) for AcmeSuite cloud platform services.

2. UPTIME GUARANTEE
2.1 AcmeSuite Enterprise: 99.9% monthly uptime (permits ~44 minutes downtime/month).
2.2 AcmeSuite Professional: 99.5% monthly uptime (permits ~3.6 hours downtime/month).
2.3 AcmeSuite Starter: 99.0% monthly uptime.
2.4 Planned maintenance windows (Sunday 2:00–4:00 AM IST) are excluded from
    uptime calculations.

3. SUPPORT RESPONSE TIMES (ENTERPRISE)
3.1 P1 Critical (system down): 15-minute acknowledgement, 4-hour resolution target.
3.2 P2 High (major feature unavailable): 1-hour acknowledgement, 8-hour resolution target.
3.3 P3 Medium (minor issue): 4-hour acknowledgement, 2-business-day resolution target.
3.4 P4 Low (enquiry): 1-business-day acknowledgement, 5-business-day resolution target.

4. SERVICE CREDITS
4.1 If uptime falls below the guarantee, customers are entitled to service credits:
    - 99.0–99.5%: 10% credit on monthly invoice.
    - 95.0–99.0%: 25% credit on monthly invoice.
    - Below 95.0%: 50% credit on monthly invoice.
4.2 Credits must be claimed within 30 days of the incident.
4.3 Credits are applied to the next invoice and may not be exchanged for cash.

5. EXCLUSIONS
5.1 Downtime caused by customer actions, third-party services, or force majeure
    events is excluded from SLA calculations.
5.2 Beta features are excluded from all SLA commitments.

6. DATA BACKUP
6.1 Customer data is backed up daily with 30-day retention.
6.2 Point-in-time recovery is available for Enterprise customers within the 30-day window.

7. DATA PORTABILITY
7.1 Customers may export all data in CSV/JSON format at any time via Settings → Export.
7.2 Upon contract termination, data is available for export for 30 days,
    after which it is securely deleted.

8. POLICY OWNER
Product — Customer Success. Contact success@acmesuite.com.
""",
    ),
]


def write_document(doc_tuple: tuple, base_dir: Path) -> None:
    doc_id, rel_path, dept, category, version, eff_date, source, access, title, content = doc_tuple

    file_path = base_dir / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")

    meta = {
        "document_id": doc_id,
        "title": title,
        "department": dept,
        "category": category,
        "version": version,
        "effective_date": eff_date,
        "source": source,
        "access_level": access,
        "file_path": str(file_path.relative_to(base_dir.parent.parent)),
        "file_format": file_path.suffix.lstrip("."),
        "dataset_version": "v2",
    }
    meta_path = base_dir / "metadata" / f"{doc_id}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  [OK] {doc_id:30s} -> {rel_path}")


def main():
    print("=" * 60)
    print("RAGOps Synthetic Document Generator -- V2 Dataset (+5 docs)")
    print("=" * 60)

    for doc in V2_DOCUMENTS:
        write_document(doc, RAW_DIR)

    # Update document_index.json to include V2 docs
    index_path = RAW_DIR / "metadata" / "document_index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    existing_ids = {d["document_id"] for d in index}
    for doc in V2_DOCUMENTS:
        doc_id, rel_path, dept, cat, ver, eff, src, acc, title, _ = doc
        if doc_id not in existing_ids:
            index.append({
                "document_id": doc_id,
                "title": title,
                "department": dept,
                "category": cat,
                "version": ver,
                "effective_date": eff,
                "file_path": f"data/raw/{rel_path}",
                "dataset_version": "v2",
            })

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print()
    print(f"V2 adds: {len(V2_DOCUMENTS)} new documents")
    print(f"Total in index: {len(index)}")
    print("V2 Dataset generation complete.")
    print()
    print("Next step for DVC versioning:")
    print("  git add data/raw/")
    print("  dvc add data/raw/")
    print("  git commit -m 'dataset: add V2 documents (+5)'")
    print("  dvc push")


if __name__ == "__main__":
    main()
